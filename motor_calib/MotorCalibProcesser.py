import pandas as pd
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt


class Simulator:
    def __init__(
        self,
        maxRpm: int,
        motorConstant: float,
        momentConstant: float,
        rotorDragConstant: float,
        rollingMomentConstant: float,
    ):
        self.maxRpm = maxRpm
        self.motorConstant = motorConstant
        self.momentConstant = momentConstant
        self.rotorDragConstant = rotorDragConstant
        self.rollingMomentConstant = rollingMomentConstant

    def simulate(self, rpm):
        thrust = np.sign(rpm) * (rpm**2) * self.motorConstant


class RowData:
    __slots__ = ["pwm", "V", "I", "P", "thrust", "torque", "efficiency", "rpm"]

    def __init__(self, data, from_row=True):
        if from_row:
            self.pwm = data.ESCsignalus
            self.V = data.VoltageV
            self.I = data.CurrentA
            self.P = data.ElectricalPowerW
            self.thrust = data.ThrustN
            self.torque = data.TorqueNm
            self.efficiency = data.PropellerMechEfficiencyNW
            self.rpm = data.MotorElectricalSpeedRPM * (
                1 if self.pwm > 1500 else -1
            )

        else:
            self.pwm = 0
            self.V = 0
            self.I = 0
            self.P = 0
            self.thrust = 0
            self.torque = 0
            self.efficiency = 0
            self.rpm = 0

    def initialize_data(self, pwm, V, I, P, thrust, torque, efficiency, rpm):
        self.pwm = pwm
        self.V = V
        self.I = I
        self.P = P
        self.thrust = thrust
        self.torque = torque
        self.efficiency = efficiency
        self.rpm = rpm


def makeDataSymmetric(row_data_list: list[RowData]):
    ## create two separate list where we replicate the data form the 'negative' and 'positive' side of the pwm signal
    positive_list = []
    negative_list = []
    for row in row_data_list:
        if row.pwm > 1500:
            positive_list.append(row)
            new_row = RowData(None, False)
            new_row.initialize_data(
                1500 - (row.pwm - 1500),
                row.V,
                row.I,
                row.P,
                -row.thrust,
                -row.torque,
                row.efficiency,
                -row.rpm,
            )
            positive_list.append(new_row)
        else:
            negative_list.append(row)
            new_row = RowData(None, False)
            new_row.initialize_data(
                1500 + (1500 - row.pwm),
                row.V,
                row.I,
                row.P,
                -row.thrust,
                -row.torque,
                row.efficiency,
                -row.rpm,
            )
            negative_list.append(new_row)

    positive_list = sorted(positive_list, key=lambda x: x.pwm)
    negative_list = sorted(negative_list, key=lambda x: x.pwm)

    return positive_list, negative_list


def filter_data_by_rpm(data: list[RowData], rpm: int):
    return [row for row in data if row.rpm != rpm]


def invert_orientation_of_the_propeller(data: list[RowData]):
    new_data = []
    for row in data:
        new_row = RowData(None, False)
        new_row.initialize_data(
            row.pwm,
            row.V,
            row.I,
            row.P,
            row.thrust,
            -row.torque,
            row.efficiency,
            row.rpm,
        )
        new_data.append(new_row)
    return new_data


# Load the CSV file
df = pd.read_csv("./StepsTestV2_2024-11-11_121751.csv")

# Sanitize column names to be valid identifiers
df.columns = df.columns.str.replace(r"[^\w]", "", regex=True)

# Print sanitized column names to verify
print(df.columns)

# Create a named tuple with sanitized column names
Row = namedtuple("Row", df.columns)

# Convert each row of the DataFrame into a named tuple
data = [Row(*row) for row in df.itertuples(index=False, name=None)]
data = data[1:]  # Remove first row as it contains units
data = [RowData(row) for row in data]

neg_data, pos_data = makeDataSymmetric(data)
data = pos_data
data = filter_data_by_rpm(data, 0)
ccw_data = invert_orientation_of_the_propeller(data)

# Extract data for plots
pwm = [row.pwm for row in data]
thrust = [row.thrust for row in data]
torque = [row.torque for row in data]
rads = [row.rpm * 2 * np.pi / 60 for row in data]


# Function to fit polynomial, plot, and export equation
def plot_with_fit(
    x, y, subplot, title, xlabel, ylabel, orders, color="ro", relation_name=""
):
    plt.subplot(subplot)
    plt.title(title)
    plt.plot(x, y, color, label="Data")
    errors = []

    for order in orders:
        # Fit polynomial
        coeffs = np.polyfit(x, y, order)
        poly = np.poly1d(coeffs)
        y_fit = poly(x)

        # Plot fitted polynomial
        plt.plot(x, y_fit, label=f"Polynomial Fit (order {order})")

        # Calculate error (mean squared error)
        error = np.mean((np.array(y) - y_fit) ** 2)
        errors.append(error)

        # Export polynomial equation to terminal
        equation = " + ".join(
            [f"{coeff:.3e}*x**{i}" for i, coeff in enumerate(reversed(coeffs))]
        )
        np.save(f"{relation_name}_order_{order}_coeffs.npy", coeffs)
        print(f"{relation_name} (Order {order}): y = {equation}")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    return errors


# Define polynomial orders to fit
orders = [1, 2, 3]

# Plotting data and polynomial fits
plt.figure(1)
errors_pwm_thrust = plot_with_fit(
    pwm,
    thrust,
    221,
    "PWM vs Thrust",
    "PWM",
    "Thrust (N)",
    orders,
    relation_name="PWM vs Thrust",
)
errors_pwm_torque = plot_with_fit(
    pwm,
    torque,
    223,
    "PWM vs Torque",
    "PWM",
    "Torque (Nm)",
    orders,
    relation_name="PWM vs Torque",
)
errors_rpm_thrust = plot_with_fit(
    rads,
    thrust,
    222,
    "rads vs Thrust",
    "rads",
    "Thrust (N)",
    orders,
    relation_name="RPM vs Thrust",
)
errors_rpm_torque = plot_with_fit(
    rads,
    torque,
    224,
    "rads vs Torque",
    "rads",
    "Torque (Nm)",
    orders,
    relation_name="RPM vs Torque",
)

plt.subplots_adjust(hspace=0.5, wspace=0.5)

plt.figure(2)
errors_pwm_rpm = plot_with_fit(
    pwm,
    rads,
    111,
    "rads vs PWM",
    "PWM",
    "rads",
    orders,
    "ro",
    relation_name="RPM vs PWM",
)

# Plotting errors for each fit
plt.figure(3)
x_labels = ["Order 1", "Order 2", "Order 3"]

plt.subplot(321)
plt.bar(x_labels, errors_pwm_thrust)
plt.title("Error for PWM vs Thrust")
plt.ylabel("Mean Squared Error")

plt.subplot(323)
plt.bar(x_labels, errors_pwm_torque)
plt.title("Error for PWM vs Torque")
plt.ylabel("Mean Squared Error")

plt.subplot(322)
plt.bar(x_labels, errors_rpm_thrust)
plt.title("Error for RPM vs Thrust")
plt.ylabel("Mean Squared Error")

plt.subplot(324)
plt.bar(x_labels, errors_rpm_torque)
plt.title("Error for RPM vs Torque")
plt.ylabel("Mean Squared Error")

plt.figure(3)

positive_list, negative_list = makeDataSymmetric(data)

neg_thrust = [row.thrust for row in negative_list]
neg_rpm = [row.rpm for row in negative_list]

pos_thrust = [row.thrust for row in positive_list]
pos_rpm = [row.rpm for row in positive_list]

plt.plot(neg_rpm, neg_thrust, "ro", label="Negative PWM")
plt.plot(pos_rpm, pos_thrust, "bo", label="Positive PWM")
plt.xlabel("RPM")
plt.ylabel("Thrust (N)")
plt.title("Thrust vs RPM")
plt.legend()

plt.subplot(313)
plt.bar(x_labels, errors_pwm_rpm)
plt.title("Error for RPM vs PWM")
plt.ylabel("Mean Squared Error")

plt.subplots_adjust(hspace=0.6)

plt.figure(4)
ccw_torque = [row.torque for row in ccw_data]

plt.plot(rads, torque, "ro", label="CW Torque")
plt.plot(rads, ccw_torque, "bo", label="CCW Torque")

plt.figure(5)
plot_with_fit(
    rads,
    torque,
    111,
    "rads vs Torque",
    "rads",
    "Torque (Nm)",
    [3],
    "ro",
    relation_name="CW Torque",
)
plt.figure(6)
plot_with_fit(
    rads,
    ccw_torque,
    111,
    "rads vs Torque",
    "rads",
    "Torque (Nm)",
    [3],
    "bo",
    relation_name="CCW Torque",
)
plt.figure(7)
plot_with_fit(
    rads,
    thrust,
    111,
    "rads vs Thrust",
    "rads",
    "Thrust (N)",
    [3],
    "ro",
    relation_name="CW Thrust",
)

plt.show()
