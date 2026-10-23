#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_TSL2591.h>
#include "model.h"

Adafruit_TSL2591 tsl = Adafruit_TSL2591(2591);

// Hardware
const int UV_PIN = 34;

const int SDA_PIN = 21;
const int SCL_PIN = 22;

// Same filter used during data collection
const float ALPHA = 0.2f;

// About 10 Hz
const unsigned long SAMPLE_INTERVAL_MS = 100;

// EMA state
float uvFiltered = -1.0f;
float irFiltered = -1.0f;

unsigned long lastSampleTime = 0;


void configureTSL2591() {
  tsl.setGain(TSL2591_GAIN_MED);
  tsl.setTiming(TSL2591_INTEGRATIONTIME_100MS);
}


float scaleValue(float value, float minimum, float maximum) {
  if (maximum <= minimum) {
    return 0.0f;
  }

  float scaled = (value - minimum) / (maximum - minimum);

  if (scaled < 0.0f) {
    scaled = 0.0f;
  }

  if (scaled > 1.0f) {
    scaled = 1.0f;
  }

  return scaled;
}


int getGridIndex(float scaledValue) {
  int index = (int)(scaledValue * GRID_N);

  if (index < 0) {
    index = 0;
  }

  if (index >= GRID_N) {
    index = GRID_N - 1;
  }

  return index;
}


void setup() {
  Serial.begin(115200);

  analogReadResolution(12);

  Wire.begin(SDA_PIN, SCL_PIN);

  if (!tsl.begin()) {
    Serial.println("ERROR: TSL2591 not found.");
    while (1) {
      delay(1000);
    }
  }

  configureTSL2591();

  Serial.println("On-device classifier ready.");
  Serial.println();
}


void loop() {
  unsigned long now = millis();

  if (now - lastSampleTime < SAMPLE_INTERVAL_MS) {
    return;
  }

  lastSampleTime = now;

  unsigned long inferenceStart = micros();


  // --------------------------------------------------
  // 1. Read sensors
  // --------------------------------------------------

  int rawUV = analogRead(UV_PIN);

  uint32_t luminosity = tsl.getFullLuminosity();

  uint16_t full = luminosity & 0xFFFF;
  uint16_t ir = luminosity >> 16;

  if (full == 0xFFFF || ir == 0xFFFF) {
    Serial.println("WARNING: TSL2591 saturated.");
  }

  int rawIR = ir;


  // --------------------------------------------------
  // 2. Apply the same EMA filter used during collection
  // --------------------------------------------------

  if (uvFiltered < 0.0f) {
    uvFiltered = rawUV;
    irFiltered = rawIR;
  } else {
    uvFiltered =
      ALPHA * rawUV +
      (1.0f - ALPHA) * uvFiltered;

    irFiltered =
      ALPHA * rawIR +
      (1.0f - ALPHA) * irFiltered;
  }


  // --------------------------------------------------
  // 3. Apply the same log1p transform used during training
  // --------------------------------------------------

  float uvLog = log1p(uvFiltered);
  float irLog = log1p(irFiltered);


  // --------------------------------------------------
  // 4. Scale using the training-set bounds from model.h
  // SCALE_BOUNDS:
  // [uv_min, uv_max, ir_min, ir_max]
  // --------------------------------------------------

  float uvScaled = scaleValue(
    uvLog,
    SCALE_BOUNDS[0],
    SCALE_BOUNDS[1]
  );

  float irScaled = scaleValue(
    irLog,
    SCALE_BOUNDS[2],
    SCALE_BOUNDS[3]
  );


  // --------------------------------------------------
  // 5. Convert scaled values into lookup-table cell
  // --------------------------------------------------

  int col = getGridIndex(uvScaled);
  int row = getGridIndex(irScaled);


  // --------------------------------------------------
  // 6. Read one byte from each class table
  // --------------------------------------------------

  uint8_t classValues[NUM_CLASSES];

  int bestClass = 0;
  uint8_t bestValue = 0;

  for (int c = 0; c < NUM_CLASSES; c++) {

    uint8_t value = pgm_read_byte(
      &CLASS_TABLE[c][row][col]
    );

    classValues[c] = value;

    if (value > bestValue) {
      bestValue = value;
      bestClass = c;
    }
  }


  // --------------------------------------------------
  // 7. Apply reject threshold
  // --------------------------------------------------

  float bestNormalized =
    bestValue / 255.0f;

  bool rejected =
    bestNormalized < REJECT_TAU;


  unsigned long inferenceEnd = micros();

  unsigned long inferenceTime =
    inferenceEnd - inferenceStart;


  // --------------------------------------------------
  // 8. Report result
  // --------------------------------------------------

  Serial.print("rawUV=");
  Serial.print(rawUV);

  Serial.print(", rawIR=");
  Serial.print(rawIR);

  Serial.print(", filteredUV=");
  Serial.print(uvFiltered, 2);

  Serial.print(", filteredIR=");
  Serial.print(irFiltered, 2);

  Serial.print(", row=");
  Serial.print(row);

  Serial.print(", col=");
  Serial.print(col);

  Serial.print(", bytes=[");

  for (int c = 0; c < NUM_CLASSES; c++) {

    Serial.print(classValues[c]);

    if (c < NUM_CLASSES - 1) {
      Serial.print(",");
    }
  }

  Serial.print("], prediction=");

  if (rejected) {
    Serial.print("UNKNOWN");
  } else {
    Serial.print(CLASS_NAMES[bestClass]);
  }

  Serial.print(", time_us=");
  Serial.println(inferenceTime);
}