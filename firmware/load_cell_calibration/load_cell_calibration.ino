#include <HX711.h>   // RobTillaart/HX711 library


// Choose two free GPIOs on RAK11300/WisBlock for HX711
#define HX711_DOUT_PIN WB_IO3    // DT / DOUT from HX711
#define HX711_SCK_PIN  WB_IO4    // SCK / CLK from HX711


// Known calibration mass in grams
const float KNOWN_MASS_G = 460.0;


HX711 scale;
String inputLine;


// Persistent calibration state (must be globals, not re-initialized
// locals inside handleCommand, or every command call would reset them)
bool cal_set = false;
bool fastProcessor = false;

// Explicit forward declaration. This makes the build independent of
// Arduino's auto-prototype (ctags) step, which can silently fail to
// generate a prototype if anything earlier in the file has unbalanced
// braces or stray statements outside a function body.
void handleCommand(const String &cmd);


void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("\nHX711 load-cell calibration (RAK11300, RobTillaart lib)");

  // fastProcessor = true: RP2040 can toggle GPIO faster than the HX711's
  // minimum clock pulse spec (~0.2 us), so this inserts a small delay
  // in shiftIn() to keep timing within datasheet limits.
  scale.begin(HX711_DOUT_PIN, HX711_SCK_PIN, fastProcessor=true);

  if (!scale.wait_ready_timeout(1000)) {
    Serial.println("ERROR: HX711 not ready. Check wiring and 3.3 V supply.");
    while (true) {
      delay(1000);
    }
  }

  // Start with raw scale factor = 1 until calibrated
  scale.set_scale(1.0);

  Serial.println("Remove all load from the load cell.");
  Serial.println("Taring in 3 seconds...");
  delay(3000);

  scale.tare(20);  // Average 20 samples for zero offset
  Serial.print("Initial offset: ");
  Serial.println(scale.get_offset());
}

void loop() {
  if (Serial.available()) {
    inputLine = Serial.readStringUntil('\n');
    inputLine.trim();
    if (inputLine.length() > 0) {
      handleCommand(inputLine);
    }
  }
  delay(250);
}

void handleCommand(const String &cmd) {
  if (cmd == "PING") {
    Serial.println("READY");
  }
  else if (cmd == "OFFSET") {
    if (!scale.wait_ready_timeout(1000)) {
      Serial.println("HX711_NOT_READY");
    } else {
      scale.tare(20);
      Serial.println("TARE_ACK");
      Serial.print("Offset: ");
      Serial.println(scale.get_offset());
    }
  }
  else if (cmd == "SCALE") {
    if (!scale.wait_ready_timeout(1000)) {
      Serial.println("HX711_NOT_READY");
    } else {
      Serial.println("Place Weight");
      Serial.println("Scale calculation starts in 3s...");
      delay(3000);

      // calibrate_scale() re-reads 'times' raw samples, subtracts the
      // existing tare offset, and derives + applies the scale factor
      // internally in one call.
      scale.calibrate_scale(KNOWN_MASS_G, 20);
      cal_set = true;

      Serial.print("Scale: ");
      Serial.println(scale.get_scale(), 6);
    }
  }
  else if (cmd == "READWEIGHT") {
    if (!cal_set) {
      Serial.println("CAL_NOTSET");
    } else if (!scale.wait_ready_timeout(1000)) {
      Serial.println("HX711_NOT_READY");
    } else {
      // get_units() divides by the calibrated scale factor, unlike
      // get_value() which only subtracts the offset. This was the bug
      // in the previous script that produced wrong / noisy numbers.
      float weight = scale.get_units(10);
      Serial.print("WEIGHT,");
      Serial.println(weight, 4);
    }
  }
  // Unknown commands are ignored silently on purpose.
}