"""AstroSense backend (no GUI): sensor table, physics-based simulator, Isolation Forest detector."""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

# min/max = nominal band. src = where the range comes from ("ASSUMPTION" = not sourced, replace it)
FIELDS = [
    dict(key="battery_voltage", label="Battery / Bus Voltage", unit="V", group="power", min=24.5, max=29.4, hard_min=21, hard_max=31, default=28.0,
         src="28 V-class unregulated bus (NASA/TM-2003-212304; Bekhti 2022). Band derived: 7-series Li-ion pack, 3.5-4.2 V per cell."),
    dict(key="solar_output", label="Solar Array Output", unit="W", group="power", min=0, max=179, hard_min=0, hard_max=200, default=120,
         src="Small-spacecraft panel peak power 2-179 W (NASA/TP-20250000142, State-of-the-Art Small Spacecraft Technology). 0 W in eclipse."),
    dict(key="fuel_pressure", label="Propellant Tank Pressure", unit="bar", group="propulsion", min=5.5, max=24, hard_min=0, hard_max=26, default=15,
         src="Hydrazine blow-down 24 -> 5.5 bar, MEOP 24.6 bar (ArianeGroup PM25 monopropellant system datasheet)."),
    dict(key="thruster_flow", label="Thruster Mass Flow (firing)", unit="g/s", group="propulsion", min=0.13, max=0.47, hard_min=0, hard_max=0.6, default=0.3,
         src="mdot = F/(g0*Isp): 1.0 N, 215 s @ 22 bar -> 0.47 g/s; 0.25 N, 200 s @ 5.5 bar -> 0.13 g/s (PM25 datasheet figures)."),
    dict(key="engine_temp", label="Engine Core Temp", unit="C", group="propulsion", min=250, max=480, hard_min=150, hard_max=650, default=360,
         src="ASSUMPTION - not sourced. Replace with your thruster datasheet limits."),
    dict(key="avionics_temp", label="Avionics Temp", unit="C", group="avionics", min=5, max=45, hard_min=-10, hard_max=65, default=24,
         src="ASSUMPTION - not sourced. Replace with the unit's qualification temperature limits."),
    dict(key="cabin_pressure", label="Cabin Total Pressure", unit="kPa", group="cabin", min=97.9, max=102.7, hard_min=90, hard_max=106, default=100.3,
         src="ISS cabin design range 97.9-102.7 kPa (NASA NTRS 20120015343); NASA-STD-3001 Vol. 2 [V2 6006]."),
    dict(key="cabin_oxygen", label="Cabin O2 Partial Pressure", unit="kPa", group="cabin", min=19.5, max=23.1, hard_min=16, hard_max=26, default=21.3,
         src="ISS design ppO2 19.5-23.1 kPa (NASA NTRS 20120015343); NASA-STD-3001 [V2 6003]."),
    dict(key="cabin_co2", label="Cabin CO2 Partial Pressure", unit="mmHg", group="cabin", min=1, max=3, hard_min=0, hard_max=9, default=2,
         src="NASA-STD-3001 Vol. 2 [V2 6004]: 1-h average <= 3 mmHg; ISS observed 1-9 mmHg (NASA OCHMO-TB-004; Law & Alexander, NTRS 20150019624)."),
    dict(key="gyro_drift", label="Gyroscope Drift", unit="deg/s", group="attitude", min=0, max=0.05, hard_min=-0.2, hard_max=0.3, default=0.02,
         src="ASSUMPTION - not sourced. Replace with your gyro datasheet bias stability."),
    dict(key="wheel_rpm", label="Reaction Wheel Speed", unit="RPM", group="attitude", min=500, max=5000, hard_min=0, hard_max=6000, default=2500,
         src="Max speed 4500-5000 rpm for CubeSat-class wheels (CubeSpace CW40K, AAC RW400); +/-6000 rpm for large wheels (JAXA/MPC Type L-A)."),
    dict(key="radiation_flux", label="Radiation Dose Rate", unit="uSv/h", group="environment", min=8, max=25, hard_min=0, hard_max=1500, default=14,
         src="ISS effective dose ~0.2-0.5 mSv/day = 8-21 uSv/h (crew dose review, PMC10919966); South Atlantic Anomaly 50-60x higher (Pille, IAC-07)."),
]
FBY = {f["key"]: f for f in FIELDS}
FAULT_OVERRIDES = {"engine_temp": 610, "fuel_pressure": 3.0, "gyro_drift": 0.21}
ZERO_OK = {"thruster_flow"}   # a reading of 0 is normal (thruster not firing), so its lower limit is the hard minimum


def limits(key, f=None):
    """(low edge, high edge, room below, room above): nominal range and the distance from each edge to the hard limit.
    f = a learned field dict (see calibrate); None = the built-in table."""
    custom = f is not None
    f = f or FBY[key]
    width = f["max"] - f["min"]
    lo = f["hard_min"] if (key in ZERO_OK and not custom) else f["min"]
    return lo, f["max"], (lo - f["hard_min"]) or width, (f["hard_max"] - f["max"]) or width

# warning text per sensor: (reading too high, reading too low)
MSG = {
    "battery_voltage": ("Check Battery \u2013 Overvoltage", "Check Battery \u2013 Undervoltage / charge loss"),
    "solar_output": ("Check Solar Array \u2013 Abnormal output spike", "Check Solar Array \u2013 Power output drop"),
    "fuel_pressure": ("Check Fuel Tank \u2013 Over-pressure", "Check Fuel Tank \u2013 Pressure drop (possible leak)"),
    "thruster_flow": ("Check Thrusters \u2013 Excess flow", "Check Thrusters \u2013 Flow blockage"),
    "engine_temp": ("Check Engine \u2013 Overheating", "Check Engine \u2013 Abnormally cold"),
    "avionics_temp": ("Check Avionics \u2013 Overheating", "Check Avionics \u2013 Too cold"),
    "cabin_pressure": ("Check Cabin \u2013 Over-pressure", "Check Cabin \u2013 Pressure leak"),
    "cabin_oxygen": ("Check Life Support \u2013 Oxygen excess", "Check Life Support \u2013 Oxygen low"),
    "cabin_co2": ("Check Life Support \u2013 CO2 build-up", "Check CO2 sensor \u2013 reading too low"),
    "gyro_drift": ("Check Gyroscope \u2013 Excess drift", "Check Gyroscope \u2013 Negative drift"),
    "wheel_rpm": ("Check Reaction Wheel \u2013 Over-speed", "Check Reaction Wheel \u2013 Speed loss"),
    "radiation_flux": ("Check Shielding \u2013 High radiation flux", "Check Radiation sensor \u2013 Low flux"),
    "particle_density": ("Check Environment sensor \u2013 High particle density", "Check Environment sensor \u2013 Low reading"),
}

TIME_NAMES = {"timestamp", "time", "datetime", "date"}
norm = lambda s: str(s).strip().lower().replace(" ", "_")
ALIAS = {}
for _f in FIELDS:
    ALIAS[norm(_f["key"])] = _f["key"]
    ALIAS[norm(_f["label"])] = _f["key"]


def warning_text(col, z):
    hi, lo = MSG.get(col, (None, None))
    name = FBY[col]["label"] if col in FBY else col
    return (hi if z > 0 else lo) or f"Check {name} \u2013 abnormally {'high' if z > 0 else 'low'}"


# ---------------------------------------------------------------- data helpers
def read_csv(path):
    df = pd.read_csv(path)
    df.columns = [ALIAS.get(norm(c), str(c).strip()) for c in df.columns]
    return df


def time_col(df):
    return next((c for c in df.columns if str(c).lower() in TIME_NAMES), None)


def clean(df, cols):
    """Remove broken lines: non-numeric, missing or infinite values in the sensor columns."""
    n0, df = len(df), df.copy()
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=cols)
    return df, n0 - len(df)


# ---------------------------------------------------------------- learn ranges from the user's own healthy data
BAD_WORDS = ("anomal", "fault", "fail", "alert", "warn", "critical", "abnormal")
UNIT_TOKENS = {"v": "V", "w": "W", "psi": "PSI", "lps": "L/s", "c": "C", "kpa": "kPa", "pct": "%", "ppm": "ppm",
               "degpersec": "deg/s", "rpm": "RPM", "usv": "uSv"}


def unit_from_column(col, default=""):
    """'Fuel_Tank_Pressure_PSI' -> 'PSI' (unit written as the last word of the column name)."""
    return UNIT_TOKENS.get(str(col).strip().split("_")[-1].lower(), default)


def find_labels(df):
    """Look for a ground-truth column in an uploaded log (e.g. Anomaly_Flag, Overall_Status).
    Returns (Series, column name) with True = anomaly, or (None, None) when the file has no labels.
    Text columns are read by their words; a two-value number column is read as 'the rarer value = anomaly'."""
    for c in df.columns:                                              # text status column first
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            continue
        u = s.astype(str).str.strip().str.lower()
        if u.nunique() <= 6 and u.str.contains("|".join(BAD_WORDS)).any():
            return u.str.contains("|".join(BAD_WORDS)) & ~u.str.contains(r"^no\b|^not\b|^non"), c
    for c in df.columns:                                              # then a 0/1 or 1/-1 flag column
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and any(w in str(c).lower() for w in ("anomal", "flag", "label", "fault", "outlier")):
            vc = s.dropna().value_counts()
            if len(vc) == 2 and vc.iloc[0] != vc.iloc[1]:
                return s == vc.index[-1], c
    return None, None


def calibrate(df, keys, units=None, q=0.005, margin=0.15, hard=3.0):
    """Learn each sensor's nominal band from HEALTHY rows: the 0.5-99.5 percentile range widened by 15 %.
    The hard limit (0 % sensor health) is 3 band-widths beyond each edge. Returns {key: field dict}."""
    bands = {}
    for k in keys:
        lo, hi = np.quantile(df[k].to_numpy(float), [q, 1 - q])
        w = (hi - lo) or abs(hi) * 0.01 or 1.0
        lo, hi = lo - margin * w, hi + margin * w
        w = hi - lo
        bands[k] = dict(FBY[k], min=lo, max=hi, hard_min=lo - hard * w, hard_max=hi + hard * w,
                        unit=(units or {}).get(k) or FBY[k]["unit"], src="Learned from the healthy rows of the uploaded file.")
    return bands


EQUATIONS = """EQUATIONS USED
Min-Max scaling:   x' = (x - xmin) / (xmax - xmin)
Isolation Forest:  s(x) = 2^(-E[h(x)] / c(n)),  c(n) = 2H(n-1) - 2(n-1)/n   (Liu, Ting & Zhou, ICDM 2008)
Sensor health:     h_i = 1 - (distance outside nominal range) / (distance from range edge to the sensor's hard limit), clipped to 0..1
                   (h_i = 100 % anywhere inside the nominal range, 0 % at the hard limit)
Series system:     H_sensors = 100 * h_1 * h_2 * ... * h_n      (every part must work, as in reliability engineering)
Pattern health:    H_model = piecewise-linear map of s(x): 100 at the 95th percentile of healthy scores, 80 at the flag threshold, 0 at the score of a reading with every sensor at its hard limit
Record health:     H = min(H_sensors, H_model)
System health:     mean(H over all records) * (share of records NOT flagged)
Deviation:         z = (x - mean_healthy) / std_healthy
Solar power:       P = P_peak * cos(theta) * sunlit
Battery:           dSoC/dt = (P_solar - P_load) / E_cap ;  V_bus = N_s * (3.5 + 0.7 * SoC)
Tank blow-down:    P * V_gas = const  ->  P = P0 / (1 + V_used / V_gas0)
Hydrazine flow:    mdot = F / (g0 * Isp)
Cabin gas balance: dp/dt = g - lambda * p   (material balance, Perry, ICES-2019-58)
Dalton's law:      p_i = x_i * P_total
Thermal lag:       T' = (T_eq - T) / tau   (first-order, C dT/dt = Q - (T - T_env)/R)"""
G0 = 9.80665


def _lag(x, tau, dt=60.0):
    y, out = x[0], []
    for v in x:
        y += dt / tau * (v - y)
        out.append(y)
    return np.array(out)


def _ar1(rng, n, mean, sd, rho=0.95):
    e, x = rng.normal(size=n), np.zeros(n)
    for k in range(1, n):
        x[k] = rho * x[k - 1] + np.sqrt(1 - rho ** 2) * sd * e[k]
    return mean + x


def simulate(n=3000, seed=7, p_range=(23.0, 6.0), faults=()):
    """Physics-based telemetry, 1 sample/min. faults: any of 'scrubber', 'overheat', 'leak'."""
    rng = np.random.default_rng(seed)
    dt, T_ORB, idx = 60.0, 5400.0, np.arange(n)
    t, ph = idx * dt, (idx * dt / T_ORB) % 1.0
    sun = (ph < 0.62).astype(float)                                          # ~38 % of each orbit in eclipse
    ramp = lambda a, b, c: np.clip((idx - a) / (b - a), 0, 1) * (idx < c)   # rises a->b, holds until c

    # power: P = P_peak cos(theta) sunlit ; SoC' = (P_sol - P_load)/E ; V = Ns (3.5 + 0.7 SoC)
    cos_th = np.clip(0.85 + 0.1 * np.sin(2 * np.pi * t / 86400) + rng.normal(0, .02, n), 0.5, 1)
    p_sol = 170 * cos_th * sun
    soc, volt = 0.7, []
    for k in range(n):
        soc = float(np.clip(soc + (p_sol[k] - (85 + 10 * rng.normal())) * dt / 3600 / 150.0, 0.1, 1.0))
        volt.append(7 * (3.5 + 0.7 * soc) + rng.normal(0, .05))

    # propulsion: isothermal blow-down, thrust & Isp interpolated between the PM25 datasheet points
    P = p_range[0] / (1 + (p_range[0] / p_range[1] - 1) * idx / max(n - 1, 1))
    frac = (np.clip(P, 5.5, 22) - 5.5) / 16.5
    fire = rng.random(n) < 0.35
    mdot = np.where(fire, (0.25 + 0.75 * frac) / (G0 * (200 + 15 * frac)) * 1000, 0.0)   # g/s
    eng = _lag(300 + 180 * fire, 600) + rng.normal(0, 3, n) + (300 * ramp(180, 230, 260) if "overheat" in faults else 0)
    P_meas = P * (1 - (0.6 * ramp(180, 230, 260) if "leak" in faults else 0)) + rng.normal(0, .05, n)

    # cabin: CO2 balance, Dalton's law
    lam, pc, co2 = 1 / 1800, 2.0, []
    for k in range(n):
        lam_k = lam * 0.25 if ("scrubber" in faults and 80 <= k < 160) else lam
        pc += dt * (lam * 2.0 * (1 + 0.3 * np.sin(2 * np.pi * t[k] / 86400)) - lam_k * pc) + rng.normal(0, .01)
        co2.append(pc)
    ptot = _ar1(rng, n, 100.3, 0.15)
    ppo2 = (0.2125 + _ar1(rng, n, 0, 0.0008)) * ptot

    # environment: background dose + occasional South Atlantic Anomaly bump
    orbit = (t // T_ORB).astype(int)
    saa = (rng.random(orbit.max() + 1) < 0.4)[orbit] & (ph >= 0.30) & (ph < 0.40)
    rad = 14 * (1 + 0.12 * rng.normal(size=n)) * (1 + 2 * saa)

    df = pd.DataFrame({
        "battery_voltage": volt, "solar_output": p_sol + rng.normal(0, 1.5, n) * sun, "fuel_pressure": P_meas,
        "thruster_flow": mdot, "engine_temp": eng, "avionics_temp": _lag(20 + 8 * sun, 1800) + rng.normal(0, .3, n),
        "cabin_pressure": ptot, "cabin_oxygen": ppo2, "cabin_co2": co2, "gyro_drift": _ar1(rng, n, 0.02, 0.006, 0.9),
        "wheel_rpm": 2500 + 800 * np.sin(2 * np.pi * ph) + rng.normal(0, 30, n), "radiation_flux": rad})
    df.insert(0, "timestamp", pd.date_range("2026-10-04 00:00", periods=n, freq="min"))
    return df


def make_baseline(n=3000, seed=7):
    """Healthy training data: a whole propellant lifetime (23 -> 6 bar) of simulated telemetry."""
    return simulate(n, seed).drop(columns="timestamp")


def make_sample_log(n=300, seed=21):
    """Mid-life test log (14 bar) with three injected faults: CO2 scrubber, overheating, propellant leak."""
    return simulate(n, seed, p_range=(14.0, 13.5), faults=("scrubber", "overheat", "leak"))


# ---------------------------------------------------------------- the ML model
class Detector:
    z_guard = 7.0   # |z| >= 7 on any sensor also flags the record: Isolation Forest is weak on single-sensor faults
    band_floor = 0.08          # sigma of a known sensor is never below 8 % of its nominal band width (stops false alarms on very steady sensors)

    def field(self, c):
        """Range data of a sensor: learned band if the detector was calibrated on a file, else the built-in table."""
        return getattr(self, "bands", {}).get(c) or FBY.get(c)

    def fit(self, df, trees=200, contamination=0.01, bands=None):
        self.bands = bands or {}
        self.cols = list(df.columns)
        self.mean, self.std = df.mean(), df.std().replace(0, 1).fillna(1)
        for c in self.cols:                                                   # very steady sensors (e.g. cabin pressure) would
            if self.field(c):                                                 # otherwise flag normal in-band readings
                self.std[c] = max(self.std[c], self.band_floor * (self.field(c)["max"] - self.field(c)["min"]))
        self.scaler = MinMaxScaler().fit(df)                                  # Min-Max Scaling
        X = self.scaler.transform(df)
        self.model = IsolationForest(n_estimators=trees, contamination=contamination,
                                     random_state=42).fit(X)                  # learns healthy pattern
        a = -self.model.score_samples(X)                                      # anomaly score (higher = odder)
        self.a_thr = float(-self.model.offset_)                               # flag threshold
        self.a_ok = min(float(np.percentile(a, 95)), self.a_thr - 1e-6)       # still "fully healthy"
        span = df.max() - df.min()                                            # 0 % health = every sensor at its hard limit
        corners = pd.DataFrame([{c: self.field(c)["hard_min"] if self.field(c) else df[c].min() - span[c] for c in self.cols},
                                {c: self.field(c)["hard_max"] if self.field(c) else df[c].max() + span[c] for c in self.cols}])
        a_corner = float((-self.model.score_samples(self.scaler.transform(corners))).max())
        self.a_zero = a_corner if a_corner > self.a_thr + 0.05 else 1.0

    def score(self, df):
        X = self.scaler.transform(df[self.cols])
        a = -self.model.score_samples(X)
        health = np.interp(a, [self.a_ok, self.a_thr, self.a_zero], [100, 80, 0])   # per-record health 0-100
        z = (df[self.cols] - self.mean) / self.std                            # deviation from healthy mean
        guard = (z.abs().max(axis=1) >= self.z_guard).to_numpy()              # single-sensor guard (see README)
        hs = self.sensor_health(df)                                           # per-sensor health 0..1 from nominal ranges
        h_sens = 100 * hs.prod(axis=1).to_numpy()                             # series system: product of sensor healths
        health = np.minimum(health, h_sens)                                   # record health = lower of the two
        flagged = (a > self.a_thr) | guard | (h_sens < 80)
        self.last = dict(a=a, h_model=np.interp(a, [self.a_ok, self.a_thr, self.a_zero], [100, 80, 0]), hs=hs, h_sens=h_sens, health=health)
        return health, flagged, z

    def sensor_health(self, df):
        """h_i = 1 inside the nominal range, falling in a straight line to 0 at the sensor's hard limit."""
        h = pd.DataFrame(1.0, index=df.index, columns=self.cols)
        for c in self.cols:
            if self.field(c):
                lo, hi, room_lo, room_hi = limits(c, self.bands.get(c))
                x = df[c].to_numpy(float)
                out = np.where(x < lo, (lo - x) / room_lo, np.where(x > hi, (x - hi) / room_hi, 0.0))
                h[c] = np.clip(1 - out, 0, 1)
        return h

    def explain(self, flagged, z):
        """For flagged records, find which sensors deviate most -> [(sensor, records, mean_z)]."""
        zf = z[flagged].iloc[:5000]
        low = (self.last["hs"][flagged].iloc[:5000] < 0.8) if getattr(self, "last", None) else None   # sensors out of range
        cnt, tot = {}, {}
        for k, (_, r) in enumerate(zf.iterrows()):
            top = set(r.abs().sort_values(ascending=False).index[:3])
            if low is not None:
                top |= {c for c in low.columns if low.iloc[k][c]}
            for c in top:
                if abs(r[c]) >= 2 or (low is not None and low.iloc[k][c]):
                    cnt[c] = cnt.get(c, 0) + 1
                    tot[c] = tot.get(c, 0) + r[c]
        if not cnt and len(zf):
            c = zf.abs().mean().idxmax()
            cnt[c], tot[c] = len(zf), zf[c].sum()
        return sorted([(c, cnt[c], tot[c] / cnt[c]) for c in cnt], key=lambda x: -x[1])

    def results(self, df, health, flagged):
        """Per-record output table for export: health score %, status and the weakest sensor.
        Call right after score(); health / flagged are the two arrays score() returned."""
        d = self.last
        hs = d["hs"]
        weakest = np.where(hs.min(axis=1).to_numpy() < 0.8, hs.idxmin(axis=1).to_numpy(), "")
        return pd.DataFrame({
            "health_score_pct": np.round(health, 1),                 # record health 0-100 %  (lower of the two below)
            "sensor_health_pct": np.round(d["h_sens"], 1),           # from nominal ranges (series system)
            "pattern_health_pct": np.round(d["h_model"], 1),         # from the Isolation Forest anomaly score
            "status": np.where(flagged, "ANOMALY", "NOMINAL"),
            "weakest_sensor": weakest,
        }, index=df.index)

    def report(self, df, flagged, overall, source="this reading"):
        """Step-by-step working of the health % for the data last passed to score()."""
        L, n, d = [], len(df), self.last
        f = lambda v: f"{v:.1f} %"
        L.append(f"HOW THE HEALTH % WAS CALCULATED   ({source})\n")
        L.append("STEP 1 - every sensor is compared with its nominal range")
        L.append("  100 % anywhere inside the range; it falls in a straight line to 0 % at the sensor's hard limit.\n")
        known = [c for c in self.cols if self.field(c)]
        if n == 1:
            L.append(f"  {'Sensor':<27}{'Reading':>16}  {'Nominal range':<24}{'Where the reading is':<36}{'Health':>8}")
            for c in known:
                x, (lo, hi, rl, rh) = float(df[c].iloc[0]), limits(c, self.bands.get(c))
                fd = self.field(c)
                u = fd["unit"]
                where = ("inside the range" if lo <= x <= hi else
                         f"{lo - x:.3g} below (limit {rl:.3g} away)" if x < lo else f"{x - hi:.3g} above (limit {rh:.3g} away)")
                L.append(f"  {fd['label']:<27}{f'{x:.4g} {u}':>16}  {f'{lo:g} - {hi:g} {u}':<24}{where:<36}{f(100 * d['hs'][c].iloc[0]):>8}")
            parts = " x ".join(f"{d['hs'][c].iloc[0]:.3f}" for c in known)
            L.append(f"\nSTEP 2 - sensor health = product of all sensor healths (a series system: every part must work)")
            L.append(f"  {parts}\n  = {f(d['h_sens'][0])}\n")
            L.append("STEP 3 - pattern health from the Isolation Forest (is this combination of readings like healthy data?)")
            L.append(f"  anomaly score a = {d['a'][0]:.3f}   (healthy data stays below {self.a_ok:.3f}; flag threshold {self.a_thr:.3f})")
            L.append(f"  maps to {f(d['h_model'][0])}   [100 % at {self.a_ok:.3f}, 80 % at {self.a_thr:.3f}, 0 % at {self.a_zero:.3f} = every sensor at its hard limit]\n")
            L.append("STEP 4 - health = the lower of step 2 and step 3")
            L.append(f"  min({f(d['h_sens'][0])}, {f(d['h_model'][0])}) = {f(d['health'][0])}   -> gauge shows {overall} %")
        else:
            L.append(f"  Done for each of the {n} records; summary over all records:\n")
            L.append(f"  {'Sensor':<28}{'Records outside range':>22}{'Average sensor health':>24}")
            for c in known:
                lo, hi, _, _ = limits(c, self.bands.get(c))
                out = int(((df[c] < lo) | (df[c] > hi)).sum())
                L.append(f"  {self.field(c)['label']:<28}{f'{out} of {n}':>22}{f(100 * d['hs'][c].mean()):>24}")
            L.append("\nSTEP 2 - per record: sensor health = product of its sensor healths (series system)")
            L.append(f"  average over records = {f(d['h_sens'].mean())}\n")
            L.append("STEP 3 - per record: pattern health from the Isolation Forest anomaly score")
            L.append(f"  average over records = {f(d['h_model'].mean())}   [100 % at a = {self.a_ok:.3f}, 80 % at {self.a_thr:.3f}, 0 % at {self.a_zero:.3f}]\n")
            L.append("STEP 4 - record health = lower of step 2 and step 3")
            L.append(f"  average record health = {f(d['health'].mean())}\n")
            k = int(flagged.sum())
            L.append("STEP 5 - system health = average record health x share of records that are NOT flagged")
            if n < 30:
                L.append(f"  (fewer than 30 records: the plain average is used) = {f(d['health'].mean())}")
            else:
                L.append(f"  {d['health'].mean():.1f} % x {n - k}/{n} ({100 * (n - k) / n:.1f} %) = {f(d['health'].mean() * (1 - k / n))}")
            L.append(f"\n  -> gauge shows {overall} %")
        L.append("\nA record is flagged when the Isolation Forest score is above the threshold, a sensor is 7 or more standard "
                 "deviations from the healthy mean, or its sensor health is below 80 %.")
        return "\n".join(L)
