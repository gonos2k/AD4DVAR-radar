const form = document.querySelector("#experiment-form");
const deck = document.querySelector("#radar-deck");
const runButton = form.querySelector("button[type='submit']");
const runStatus = document.querySelector("#run-status");

const controls = {
  use_background: document.querySelector("#use-background"),
  background_age_minutes: document.querySelector("#background-age"),
  shift_y: document.querySelector("#shift-y"),
  shift_x: document.querySelector("#shift-x"),
  intensity_bias_dbz: document.querySelector("#intensity-bias"),
  coverage_percent: document.querySelector("#coverage"),
  lead_minutes: document.querySelector("#lead-minutes"),
};

const labels = [
  [controls.background_age_minutes, "#background-age-value", (value) => `${value}분`],
  [controls.shift_y, "#shift-y-value", signedPixels],
  [controls.shift_x, "#shift-x-value", signedPixels],
  [controls.intensity_bias_dbz, "#intensity-bias-value", (value) => `${signed(value)} dBZ`],
  [controls.coverage_percent, "#coverage-value", (value) => `${value}%`],
  [controls.lead_minutes, "#lead-minutes-value", (value) => `+${value}분`],
];

for (const [input, selector, format] of labels) {
  input.addEventListener("input", () => {
    document.querySelector(selector).textContent = format(input.value);
  });
}

controls.use_background.addEventListener("change", updateBackgroundControls);
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  await run("/api/run", settings());
});

function signed(value) {
  const number = Number(value);
  return number > 0 ? `+${number}` : `${number}`;
}

function signedPixels(value) {
  return `${signed(value)} px`;
}

function updateBackgroundControls() {
  const disabled = !controls.use_background.checked;
  for (const name of [
    "background_age_minutes",
    "shift_y",
    "shift_x",
    "intensity_bias_dbz",
    "coverage_percent",
  ]) {
    controls[name].disabled = disabled;
  }
}

function settings() {
  return {
    use_background: controls.use_background.checked,
    background_age_minutes: Number(controls.background_age_minutes.value),
    shift_y: Number(controls.shift_y.value),
    shift_x: Number(controls.shift_x.value),
    intensity_bias_dbz: Number(controls.intensity_bias_dbz.value),
    coverage_percent: Number(controls.coverage_percent.value),
    lead_minutes: Number(controls.lead_minutes.value),
  };
}

async function run(path, payload) {
  setRunning(true);
  try {
    const options = payload
      ? {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        }
      : undefined;
    const response = await fetch(path, options);
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.error || `HTTP ${response.status}`);
    }
    render(result);
    runStatus.textContent = "계산이 끝났습니다. 초기장만 바꿔 다시 비교할 수 있습니다.";
    runStatus.classList.remove("error");
  } catch (error) {
    runStatus.textContent = `계산 실패: ${error.message}`;
    runStatus.classList.add("error");
  } finally {
    setRunning(false);
  }
}

function setRunning(running) {
  deck.classList.toggle("running", running);
  deck.setAttribute("aria-busy", String(running));
  runButton.disabled = running;
  if (running) {
    runStatus.textContent = "ADVAR가 고정 사례를 다시 계산하고 있습니다.";
  }
}

function render(result) {
  document.querySelector("#case-name").textContent = result.case.name;
  document.querySelector("#case-description").textContent = result.case.description;
  document.querySelector("#background-caption").textContent = result.state.background_used
    ? `age ${result.settings.background_age_minutes}분`
    : "사용 안 함";
  document.querySelector("#forecast-caption").textContent = `+${result.case.lead_minutes}분`;
  document.querySelector("#truth-caption").textContent = `+${result.case.lead_minutes}분`;

  drawField(document.querySelector("#observation-canvas"), result.fields.observation);
  drawField(document.querySelector("#background-canvas"), result.fields.background);
  drawField(document.querySelector("#forecast-canvas"), result.fields.forecast);
  drawField(document.querySelector("#truth-canvas"), result.fields.truth);

  metric("#forecast-mae", result.metrics.forecast_mae_dbz, " dBZ");
  metric("#persistence-mae", result.metrics.persistence_mae_dbz, " dBZ");
  const skill = document.querySelector("#skill");
  metric("#skill", result.metrics.skill_dbz, " dBZ", true);
  skill.classList.toggle("positive", result.metrics.skill_dbz > 0);
  skill.classList.toggle("negative", result.metrics.skill_dbz < 0);
  percent("#valid-fraction", result.metrics.valid_fraction);
  percent("#mean-confidence", result.metrics.mean_confidence);
  percent("#background-fraction", result.metrics.background_contribution_fraction);

  const processList = document.querySelector("#process-list");
  processList.replaceChildren(
    ...result.process.map((step) => {
      const item = document.createElement("li");
      const name = document.createElement("strong");
      const detail = document.createElement("span");
      name.textContent = step.name;
      detail.textContent = step.detail;
      item.append(name, detail);
      return item;
    }),
  );
}

function metric(selector, value, suffix, showSign = false) {
  const element = document.querySelector(selector);
  if (value === null) {
    element.textContent = "유효영역 없음";
    return;
  }
  const prefix = showSign && value > 0 ? "+" : "";
  element.textContent = `${prefix}${Number(value).toFixed(2)}${suffix}`;
}

function percent(selector, value) {
  document.querySelector(selector).textContent = value === null
    ? "—"
    : `${(Number(value) * 100).toFixed(1)}%`;
}

function drawField(canvas, field) {
  const rows = field.length;
  const columns = field[0].length;
  const scale = window.devicePixelRatio || 1;
  canvas.width = columns * scale;
  canvas.height = rows * scale;
  const context = canvas.getContext("2d");
  context.scale(scale, scale);
  context.imageSmoothingEnabled = false;

  for (let y = 0; y < rows; y += 1) {
    for (let x = 0; x < columns; x += 1) {
      const value = field[y][x];
      context.fillStyle = value === null ? missingColor(x, y) : reflectivityColor(value);
      context.fillRect(x, y, 1, 1);
    }
  }
}

function missingColor(x, y) {
  return (x + y) % 4 < 2 ? "#d7e0de" : "#cbd6d4";
}

function reflectivityColor(value) {
  const stops = [
    [-10, [23, 63, 95]],
    [0, [32, 107, 138]],
    [10, [47, 163, 107]],
    [20, [217, 193, 43]],
    [30, [224, 122, 40]],
    [40, [200, 75, 75]],
    [55, [127, 60, 141]],
  ];
  const clipped = Math.max(stops[0][0], Math.min(stops.at(-1)[0], value));
  for (let index = 1; index < stops.length; index += 1) {
    const [upperValue, upperColor] = stops[index];
    const [lowerValue, lowerColor] = stops[index - 1];
    if (clipped <= upperValue) {
      const ratio = (clipped - lowerValue) / (upperValue - lowerValue);
      const color = lowerColor.map((channel, channelIndex) =>
        Math.round(channel + ratio * (upperColor[channelIndex] - channel)),
      );
      return `rgb(${color.join(",")})`;
    }
  }
  return "rgb(127,60,141)";
}

updateBackgroundControls();
run("/api/default");
