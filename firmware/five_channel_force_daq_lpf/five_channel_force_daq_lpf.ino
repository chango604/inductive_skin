/**
   @file   five_channel_force_daq_lpf.ino
   @brief  5-coil LDC1614 inductance + TAL221/HX711 force logger with a 2 Hz low-pass
           filter on dL. Used for the track-pad demonstration (Fig. 6); the
           characterization and training data were logged with five_channel_force_daq.ino.

   Serial protocol (115200 baud):
     PC -> "SAMPLE"  ARD -> one CSV data line (14 fields, 13 commas):
           millis,force_g,force_t_ms,force_seq,dL1..dL5 (uH),raw1..raw5
           dLk = Lk - L0k: CHANGE in inductance from the baseline L0 (see below).
           raw1..raw5 stay the ABSOLUTE 28-bit counts, so absolute L can be recomputed.
     PC -> "TARE"    ARD -> "# TARE_ACK"   tares the load cell AND re-captures the
                                            inductance baseline (nothing may touch the sensor)
     PC -> "ZERO"    ARD -> "# ZERO_ACK"   re-captures only the inductance baseline
     PC -> "INFO"    ARD -> "# INFO ..." lines (clock source and dividers, and the current
                                            baseline L0 of every coil), then "# INFO_END"
     Lines starting with '#' are diagnostics; the PC ignores them (except INFO, which it saves).

   Inductance baseline L0
     Average of BASELINE_READS readings per coil, captured at start-up (after the LDC
     set-up) and on every TARE / ZERO. After each capture the sketch prints
     "# BASELINE L1=... L5=..." (uH). A coil with no valid reading gets L0 = nan, and its
     dL is nan until the next successful capture.

   Coils (C = 1000 pF for all):
     L1 = 230 uH  -> ldc1 (I2C 0x2B) CH0        (centre coil)
     L2..L5 = 78 uH -> ldc2 (I2C 0x2A) CH0..CH3

   Inductance from the REAL clock settings
     The library chooses the clock dividers from the L / C constants below. At start-up
     this sketch reads them back from each chip:
       CONFIG          (0x1A) bit 9 REF_CLK_SRC: 0 = internal oscillator, 1 = external CLKIN
       CLOCK_DIVIDERSx (0x14 + ch): FIN_DIVIDER = bits 15..12, FREF_DIVIDER = bits 9..0
     and converts every reading with
       f_sensor = FIN_DIVIDER * (f_CLK / FREF_DIVIDER) * raw / 2^28
       L        = 1 / ((2 pi f_sensor)^2 * C)
     f_CLK = FCLK_EXTERNAL_HZ (40 MHz option on the RAK12029) or FCLK_INTERNAL_HZ
     (LDC1614 internal oscillator, 43.4 MHz typical). The internal oscillator of an
     individual chip can be a few % off: that scales every L by the same constant
     (dL / L0 is unaffected). The raw counts are logged as well, so L can be
     recomputed later with a measured f_CLK.

   Force: the HX711 (10 SPS) is read NON-BLOCKING in loop() and cached with its
   millis() time and a counter (force_seq), so a SAMPLE never waits for it.

   Low-pass filter on dL (this version)
     The coils are now read continuously in loop() every COIL_READ_PERIOD_MS, and each
     coil's inductance goes through a first-order low-pass filter (exponential moving
     average) with cutoff LPF_CUTOFF_HZ. The filter uses the real time between reads,
     so the cutoff does not depend on how often the PC sends SAMPLE.
     A SAMPLE reports dLk = Lk_filtered - L0k. raw1..raw5 are still the latest
     UNFILTERED counts. Set LPF_CUTOFF_HZ to 0 to turn the filter off.
     Spike rejection: a reading with |L - L0| > DL_SPIKE_LIMIT_UH is dropped before the
     filter (one bad read would otherwise smear into many samples).
     If a coil has had no accepted reading for COIL_STALE_MS, its dL is reported as nan.
     The filter is reset to L0 on every baseline capture (start-up, TARE, ZERO).
**/

#include "RAK12029_LDC1614.h"
#include <HX711.h>        // bogde/HX711 -- install via Library Manager
#include <math.h>

// ─── HX711 (TAL221) ───────────────────────────────────────────────
#define HX711_DOUT_PIN   WB_IO3
#define HX711_SCK_PIN    WB_IO4
#define LOAD_CELL_SCALE  1043.522339   // counts per gram (from your calibration)
#define LOAD_CELL_OFFSET -82979
#define TARE_READS       10             // conversions averaged by "TARE" (~1 s at 10 SPS)

// ─── Inductance baseline ──────────────────────────────────────────
#define BASELINE_READS    50            // readings averaged per coil for L0
#define BASELINE_DELAY_MS 20            // between baseline readings (~1 s in total)

// ─── dL low-pass filter ───────────────────────────────────────────
#define LPF_CUTOFF_HZ       2.0         // -3 dB cutoff; 0 = filter off
#define COIL_READ_PERIOD_MS 10          // background coil read period (~100 Hz)
#define DL_SPIKE_LIMIT_UH   10.0        // reject reads with |L - L0| above this (real dL < ~2 uH)
#define COIL_STALE_MS       250         // report nan if no accepted read for this long

// ─── Coil parameters ──────────────────────────────────────────────
#define L1_UH   230.0
#define C1_PF   1000.0
#define L2_UH   78.0
#define C2_PF   1000.0
#define L3_UH   78.0
#define C3_PF   1000.0
#define L4_UH   78.0
#define C4_PF   1000.0
#define L5_UH   78.0
#define C5_PF   1000.0
#define L_DUMMY 78.0
#define C_DUMMY 1000.0

// ─── Reference clock ──────────────────────────────────────────────
#define FCLK_INTERNAL_HZ 43.4e6   // LDC1614 internal oscillator (typical)
#define FCLK_EXTERNAL_HZ 40.0e6   // RAK12029 optional external oscillator

#define REG_CLOCK_DIVIDERS0 0x14  // + channel
#define REG_CONFIG          0x1A
#define MODULE1_ADDR   0x2B
#define MODULE2_ADDR   0x2A

RAK12029_LDC1614_Inductive ldc1(MODULE1_ADDR);   // L1
RAK12029_LDC1614_Inductive ldc2(MODULE2_ADDR);   // L2, L3, L4, L5
HX711 scale;

// Per coil: which board / channel, capacitance, clock settings read at start-up
struct CoilClock {
  RAK12029_LDC1614_Inductive *ldc;
  uint8_t ch;
  float   C_pF;
  float   L_nom;
  bool    extClk;
  uint16_t finDiv;
  uint16_t frefDiv;
  double  fclk;
};
CoilClock coil[5] = {
  {&ldc1, 0, C1_PF, L1_UH, false, 0, 0, FCLK_INTERNAL_HZ},
  {&ldc2, 0, C2_PF, L2_UH, false, 0, 0, FCLK_INTERNAL_HZ},
  {&ldc2, 1, C3_PF, L3_UH, false, 0, 0, FCLK_INTERNAL_HZ},
  {&ldc2, 2, C4_PF, L4_UH, false, 0, 0, FCLK_INTERNAL_HZ},
  {&ldc2, 3, C5_PF, L5_UH, false, 0, 0, FCLK_INTERNAL_HZ},
};

// Baseline inductance of every coil (uH); NAN until captured
float L0[5] = {NAN, NAN, NAN, NAN, NAN};

// ─── Low-pass filter state ────────────────────────────────────────
float         Lf[5]          = {NAN, NAN, NAN, NAN, NAN};  // filtered inductance (uH)
unsigned long lastAccept_us[5] = {0, 0, 0, 0, 0};          // time of last accepted read
bool          everAccepted[5]  = {false, false, false, false, false};
uint32_t      lastRaw[5]     = {0, 0, 0, 0, 0};            // latest unfiltered counts
bool          lastB1ok = true,  lastB2ok = true;    // boards checked in setup()
unsigned long lastCoilRead_ms = 0;

// ─── Force cache ──────────────────────────────────────────────────
float         lastForce_g  = NAN;
unsigned long lastForce_ms = 0;
unsigned long forceSeq     = 0;

void updateForceCache()
{
  if (scale.is_ready()) {
    lastForce_g  = scale.get_units(1);   // data is ready, so this returns at once
    lastForce_ms = millis();
    forceSeq++;
  }
}

// Read CONFIG and CLOCK_DIVIDERS of every coil's channel (after the library config)
void readClockSettings()
{
  for (int i = 0; i < 5; i++) {
    uint16_t cfg = 0, div = 0;
    coil[i].ldc->IIC_read_16bit(REG_CONFIG, &cfg);
    coil[i].ldc->IIC_read_16bit(REG_CLOCK_DIVIDERS0 + coil[i].ch, &div);
    coil[i].extClk  = (cfg >> 9) & 0x1;
    coil[i].fclk    = coil[i].extClk ? FCLK_EXTERNAL_HZ : FCLK_INTERNAL_HZ;
    coil[i].finDiv  = (div >> 12) & 0x000F;
    coil[i].frefDiv = div & 0x03FF;
  }
}

void printInfo()
{
  for (int i = 0; i < 5; i++) {
    Serial.print("# INFO L"); Serial.print(i + 1);
    Serial.print(" ch="); Serial.print(coil[i].ch);
    Serial.print(" clk="); Serial.print(coil[i].extClk ? "external" : "internal");
    Serial.print(" fclk_hz="); Serial.print(coil[i].fclk, 0);
    Serial.print(" fin_div="); Serial.print(coil[i].finDiv);
    Serial.print(" fref_div="); Serial.print(coil[i].frefDiv);
    Serial.print(" C_pF="); Serial.print(coil[i].C_pF, 1);
    Serial.print(" L_nominal_uH="); Serial.println(coil[i].L_nom, 1);
  }
  Serial.print("# INFO L0_uH");
  for (int i = 0; i < 5; i++) {
    Serial.print(" L"); Serial.print(i + 1); Serial.print("=");
    if (!isnan(L0[i])) Serial.print(L0[i], 4); else Serial.print("nan");
  }
  Serial.println();
  Serial.print("# INFO LPF cutoff_hz="); Serial.print(LPF_CUTOFF_HZ, 2);
  Serial.print(" read_period_ms="); Serial.print(COIL_READ_PERIOD_MS);
  Serial.print(" spike_limit_uH="); Serial.print(DL_SPIKE_LIMIT_UH, 2);
  Serial.print(" stale_ms="); Serial.println(COIL_STALE_MS);
  Serial.println("# INFO_END");
}

// Channel read through the library's own channel constants (works whether they
// are #defines or an enum)
int readChannel(RAK12029_LDC1614_Inductive *ldc, uint8_t ch, uint32_t *r)
{
  switch (ch) {
    case 0:  return ldc->LDC1614_get_channel_result(LDC1614_CHANNEL_0, r);
    case 1:  return ldc->LDC1614_get_channel_result(LDC1614_CHANNEL_1, r);
    case 2:  return ldc->LDC1614_get_channel_result(LDC1614_CHANNEL_2, r);
    default: return ldc->LDC1614_get_channel_result(LDC1614_CHANNEL_3, r);
  }
}

// raw (28-bit) -> inductance (uH) with the real dividers
float rawToInductance(uint32_t raw, const CoilClock &c)
{
  if (raw == 0 || c.finDiv == 0 || c.frefDiv == 0) return NAN;
  double fref  = c.fclk / (double)c.frefDiv;
  double freq  = (double)c.finDiv * fref * ((double)raw / 268435456.0);
  double omega = 2.0 * M_PI * freq;
  return (float)((1.0 / (omega * omega * (double)c.C_pF * 1e-12)) * 1e6);
}

// Read all five coils once: absolute L (uH) and raw counts. Returns the board status.
void readCoils(float L[5], uint32_t raw[5], bool &b1ok, bool &b2ok)
{
  for (int i = 0; i < 5; i++) { L[i] = NAN; raw[i] = 0; }
  uint16_t devID = 0;
  ldc1.IIC_read_16bit(LDC1614_READ_DEVICE_ID, &devID);
  b1ok = (devID == 0x3055);
  devID = 0;
  ldc2.IIC_read_16bit(LDC1614_READ_DEVICE_ID, &devID);
  b2ok = (devID == 0x3055);

  for (int i = 0; i < 5; i++) {
    bool ok = (coil[i].ldc == &ldc1) ? b1ok : b2ok;
    if (!ok) continue;
    uint32_t r = 0;
    if (readChannel(coil[i].ldc, coil[i].ch, &r) == 0) {
      raw[i] = r & 0x0FFFFFFF;                 // 28-bit conversion result
      L[i]   = rawToInductance(raw[i], coil[i]);
    }
  }
}

// Reset every coil's filter to its baseline (called after each baseline capture)
void resetCoilFilter()
{
  unsigned long now_us = micros();
  for (int i = 0; i < 5; i++) {
    Lf[i]            = L0[i];
    lastAccept_us[i] = now_us;
    everAccepted[i]  = !isnan(L0[i]);
  }
  lastCoilRead_ms = millis();
}

// Read the coils if COIL_READ_PERIOD_MS has passed and update the low-pass filters
void updateCoilFilter()
{
  unsigned long now_ms = millis();
  if ((now_ms - lastCoilRead_ms) < COIL_READ_PERIOD_MS) return;
  lastCoilRead_ms = now_ms;

  float    L[5];
  uint32_t raw[5];
  readCoils(L, raw, lastB1ok, lastB2ok);
  unsigned long now_us = micros();

  for (int i = 0; i < 5; i++) {
    lastRaw[i] = raw[i];
    if (isnan(L[i])) continue;                                   // failed read
    if (!isnan(L0[i]) && fabsf(L[i] - L0[i]) > DL_SPIKE_LIMIT_UH) continue;  // spike

    if (LPF_CUTOFF_HZ <= 0 || isnan(Lf[i]) || !everAccepted[i]) {
      Lf[i] = L[i];                                              // filter off / first value
    } else {
      float dt    = (now_us - lastAccept_us[i]) * 1e-6f;         // s since last accepted read
      float alpha = 1.0f - expf(-2.0f * (float)M_PI * LPF_CUTOFF_HZ * dt);
      Lf[i] += alpha * (L[i] - Lf[i]);
    }
    lastAccept_us[i] = now_us;
    everAccepted[i]  = true;
  }
}

// Average BASELINE_READS readings per coil into L0 (nothing may touch the sensor)
void captureBaseline()
{
  double   sum[5] = {0, 0, 0, 0, 0};
  uint16_t n[5]   = {0, 0, 0, 0, 0};
  float    L[5];
  uint32_t raw[5];
  bool b1ok, b2ok;
  for (int k = 0; k < BASELINE_READS; k++) {
    readCoils(L, raw, b1ok, b2ok);
    for (int i = 0; i < 5; i++) {
      if (!isnan(L[i])) { sum[i] += L[i]; n[i]++; }
    }
    updateForceCache();                       // keep the force cache going meanwhile
    delay(BASELINE_DELAY_MS);
  }
  Serial.print("# BASELINE");
  for (int i = 0; i < 5; i++) {
    L0[i] = (n[i] > 0) ? (float)(sum[i] / n[i]) : NAN;
    Serial.print(" L"); Serial.print(i + 1); Serial.print("=");
    if (!isnan(L0[i])) Serial.print(L0[i], 4); else Serial.print("nan");
  }
  Serial.println(" uH");
  resetCoilFilter();                          // filter restarts from the new baseline
}

void setup()
{
  pinMode(WB_IO2, OUTPUT);          // power the WisBlock slot
  digitalWrite(WB_IO2, HIGH);
  delay(300);

  pinMode(WB_IO5, OUTPUT);
  digitalWrite(WB_IO5, LOW);

  pinMode(LED_BLUE, OUTPUT);
  digitalWrite(LED_BLUE, LOW);

  Serial.begin(115200);
  Serial.setTimeout(20);
  unsigned long t0 = millis();
  while (!Serial) {
    if ((millis() - t0) > 5000) break;
    delay(100);
  }

  scale.begin(HX711_DOUT_PIN, HX711_SCK_PIN);
  scale.set_scale(LOAD_CELL_SCALE);
  scale.set_offset(LOAD_CELL_OFFSET);
  scale.tare(TARE_READS);
  Serial.println("# HX711 tared");

  ldc1.LDC1614_init();
  if (ldc1.LDC1614_multiple_channel_config(L1_UH, C1_PF,
                                           L_DUMMY, C_DUMMY,
                                           L_DUMMY, C_DUMMY,
                                           L_DUMMY, C_DUMMY) != 0) {
    Serial.println("# ERROR: LDC board 1 (0x2B) not found!");
    while (1) delay(100);
  }

  ldc2.LDC1614_init();
  if (ldc2.LDC1614_multiple_channel_config(L2_UH, C2_PF,
                                           L3_UH, C3_PF,
                                           L4_UH, C4_PF,
                                           L5_UH, C5_PF) != 0) {
    Serial.println("# ERROR: LDC board 2 (0x2A) not found!");
    while (1) delay(100);
  }

  readClockSettings();
  captureBaseline();                          // nothing may touch the sensor at power-up
  printInfo();
  Serial.println("# READY");
  Serial.println("# millis_ms,force_g,force_t_ms,force_seq,dL1..dL5_uH,raw1..raw5");
}

void loop()
{
  updateForceCache();
  updateCoilFilter();

  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd == "SAMPLE") {
      sampleAndSend();
    }
    else if (cmd == "TARE") {
      scale.tare(TARE_READS);
      lastForce_g = NAN;
      captureBaseline();                      // zero the inductance at the same moment
      Serial.println("# TARE_ACK");
    }
    else if (cmd == "ZERO") {
      captureBaseline();
      Serial.println("# ZERO_ACK");
    }
    else if (cmd == "INFO") {
      readClockSettings();
      printInfo();
    }
  }
}

void sampleAndSend()
{
  updateForceCache();
  updateCoilFilter();
  unsigned long t_ms   = millis();
  unsigned long now_us = micros();

  // Data row: millis,force_g,force_t_ms,force_seq,dL1..dL5,raw1..raw5
  Serial.print(t_ms);
  Serial.print(",");
  if (!isnan(lastForce_g)) Serial.print(lastForce_g, 3); else Serial.print("nan");
  Serial.print(",");
  Serial.print(lastForce_ms);
  Serial.print(",");
  Serial.print(forceSeq);
  for (int i = 0; i < 5; i++) {
    Serial.print(",");
    bool stale = !everAccepted[i] ||
                 (now_us - lastAccept_us[i]) > (unsigned long)COIL_STALE_MS * 1000UL;
    float dL = stale ? NAN : (Lf[i] - L0[i]);   // filtered; nan if stale or no baseline
    if (!isnan(dL)) Serial.print(dL, 4); else Serial.print("nan");
  }
  for (int i = 0; i < 5; i++) {
    Serial.print(",");
    Serial.print(lastRaw[i]);                    // latest unfiltered counts
  }
  Serial.println();

  if (!lastB1ok) Serial.println("# WARN: board 1 I2C fail");
  if (!lastB2ok) Serial.println("# WARN: board 2 I2C fail");
}
