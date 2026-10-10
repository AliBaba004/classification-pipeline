import numpy as np


SURFACES_FILE = "../assignment1/surfaces.npz"
BOUNDS_FILE = "../assignment1/scaling_bounds.npz"
OUTPUT_FILE = "../assignment1/model.h"

# Keep the same reject threshold used by the previous pipeline for now.
# We will evaluate its behaviour with the real held-out data in Part 4.
REJECT_TAU = 0.12


# Load the stored KDE surfaces
surface_data = np.load(
    SURFACES_FILE,
    allow_pickle=True
)

byte_tables = surface_data["byte_tables"]
class_names = surface_data["class_names"]
grid_n = int(surface_data["grid_n"])


# Load the training scaling bounds.
# These bounds are for the log1p-transformed sensor values.
bounds_data = np.load(BOUNDS_FILE)

minimum = bounds_data["minimum"]
maximum = bounds_data["maximum"]

xmin = float(minimum[0])
xmax = float(maximum[0])
ymin = float(minimum[1])
ymax = float(maximum[1])


num_classes = len(class_names)


def table_to_cpp(table):
    """Convert one 2D uint8 table into C++ initializer text."""

    rows = []

    for row in table:
        values = ", ".join(
            str(int(value))
            for value in row
        )

        rows.append(
            "        {" + values + "}"
        )

    return ",\n".join(rows)


with open(OUTPUT_FILE, "w", encoding="utf-8") as model_file:

    model_file.write(
        "#ifndef MODEL_H\n"
    )

    model_file.write(
        "#define MODEL_H\n\n"
    )

    model_file.write(
        "#include <Arduino.h>\n\n"
    )


    # Grid and class information
    model_file.write(
        f"#define GRID_N {grid_n}\n"
    )

    model_file.write(
        f"#define NUM_CLASSES {num_classes}\n"
    )

    model_file.write(
        f"#define REJECT_TAU {REJECT_TAU}f\n\n"
    )


    # Scaling bounds
    model_file.write(
        "// Bounds of the log1p-transformed training data\n"
    )

    model_file.write(
        "const float SCALE_BOUNDS[4] = {"
        f"{xmin:.8f}f, "
        f"{xmax:.8f}f, "
        f"{ymin:.8f}f, "
        f"{ymax:.8f}f"
        "};\n\n"
    )


    # Class names
    model_file.write(
        "const char* CLASS_NAMES[NUM_CLASSES] = {\n"
    )

    for index, name in enumerate(class_names):

        comma = "," if index < num_classes - 1 else ""

        model_file.write(
            f'    "{name}"{comma}\n'
        )

    model_file.write(
        "};\n\n"
    )


    # Byte lookup tables
    model_file.write(
        "const uint8_t "
        "CLASS_TABLE[NUM_CLASSES][GRID_N][GRID_N] "
        "PROGMEM = {\n"
    )

    for class_index in range(num_classes):

        model_file.write(
            "    {\n"
        )

        model_file.write(
            table_to_cpp(
                byte_tables[class_index]
            )
        )

        model_file.write(
            "\n    }"
        )

        if class_index < num_classes - 1:
            model_file.write(",")

        model_file.write("\n")

    model_file.write(
        "};\n\n"
    )

    model_file.write(
        "#endif\n"
    )


# Confirm the expected storage size.
table_bytes = (
    num_classes
    * grid_n
    * grid_n
)

bounds_bytes = 4 * 4


print("Generated:", OUTPUT_FILE)
print()

print("Model information")
print("Classes:", num_classes)
print("Grid:", grid_n, "x", grid_n)

print()
print("Class mapping:")

for index, name in enumerate(class_names):
    print(index, "->", name)

print()
print("Predicted storage")
print("Class tables:", table_bytes, "bytes")
print("Scaling bounds:", bounds_bytes, "bytes")
print(
    "Tables + bounds:",
    table_bytes + bounds_bytes,
    "bytes"
)