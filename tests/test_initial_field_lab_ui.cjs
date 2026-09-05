const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const test = require('node:test');

const appPath = path.resolve(__dirname, '../examples/initial_field_lab/app.js');
const initialSettings = {
  use_background: true,
  background_age_minutes: 10,
  shift_y: 0,
  shift_x: 0,
  intensity_bias_dbz: 0,
  coverage_percent: 100,
  lead_minutes: 30,
};

function response(settings, reference) {
  return {
    settings,
    case: { name: 'fixed case', description: '', lead_minutes: settings.lead_minutes },
    state: { background_used: settings.use_background },
    fields: { observation: [[0]], background: [[0]], forecast: [[0]], truth: [[0]] },
    metrics: {
      forecast_mae_dbz: 1, persistence_mae_dbz: 2, skill_dbz: 1,
      valid_fraction: 1, mean_confidence: 1, background_contribution_fraction: 1,
      scored_pixels: 1, scored_fraction: 1,
    },
    process: [],
    comparison: reference && {
      settings: reference,
      reference: { forecast_mae_dbz: 1, persistence_mae_dbz: 2 },
      candidate: { forecast_mae_dbz: 1, missing_pixels: 0 },
      improvement_dbz: 0, domain_pixels: 1, domain_fraction: 1,
    },
  };
}

function createHarness() {
  const elements = new Map();
  const requests = [];
  let failNextRequest = false;
  function element(selector) {
    if (!elements.has(selector)) {
      elements.set(selector, {
        value: '0', checked: true, textContent: '', disabled: false, hidden: false,
        handlers: {}, classList: { toggle() {}, add() {}, remove() {} },
        addEventListener(event, handler) { this.handlers[event] = handler; },
        querySelector: element, setAttribute() {}, replaceChildren() {}, append() {},
        getContext() { return { scale() {}, fillRect() {} }; },
      });
    }
    return elements.get(selector);
  }
  for (const [selector, value] of [
    ['#background-age', '10'], ['#coverage', '100'], ['#lead-minutes', '30'],
  ]) element(selector).value = value;
  const context = vm.createContext({
    document: { querySelector: element, createElement: element },
    window: { devicePixelRatio: 1 },
    fetch: async (url, options) => {
      const payload = options ? JSON.parse(options.body) : null;
      requests.push({ url, payload });
      if (failNextRequest) {
        failNextRequest = false;
        throw new Error('simulated network failure');
      }
      const { reference = null, ...settings } = payload || initialSettings;
      return { ok: true, json: async () => response(settings, reference) };
    },
  });
  vm.runInContext(fs.readFileSync(appPath, 'utf8'), context, { filename: appPath });
  return {
    element,
    requests,
    failNext() { failNextRequest = true; },
    async click(selector) { await element(selector).handlers.click(); },
    async submit() {
      await element('#experiment-form').handlers.submit({ preventDefault() {} });
    },
  };
}

test('failed repin keeps the displayed A and sends it in the next comparison', async () => {
  const harness = createHarness();
  // Drain the initial /api/default fetch and render before clicking buttons.
  await new Promise(setImmediate);
  await harness.click('#pin-reference');
  const referenceDescription = harness.element('#reference-description').textContent;
  assert.match(referenceDescription, /강도 0 dBZ/);

  harness.element('#intensity-bias').value = '6';
  await harness.submit();
  assert.equal(harness.requests.at(-1).payload.intensity_bias_dbz, 6);
  assert.deepEqual(harness.requests.at(-1).payload.reference, initialSettings);

  harness.failNext();
  await harness.click('#pin-reference');
  assert.match(harness.element('#run-status').textContent, /계산 실패/);
  assert.equal(harness.element('#reference-description').textContent, referenceDescription);

  harness.element('#intensity-bias').value = '-6';
  await harness.submit();
  assert.equal(harness.requests.at(-1).payload.intensity_bias_dbz, -6);
  assert.deepEqual(harness.requests.at(-1).payload.reference, initialSettings);
  assert.equal(harness.element('#reference-description').textContent, referenceDescription);
});

test('pin uses the calculated result and clear starts a new comparison', async () => {
  const harness = createHarness();
  await new Promise(setImmediate);
  harness.element('#intensity-bias').value = '6';
  await harness.click('#pin-reference');
  assert.equal(harness.requests.at(-1).payload.reference.intensity_bias_dbz, 0);
  await harness.submit();
  await harness.click('#pin-reference');
  assert.equal(harness.requests.at(-1).payload.reference.intensity_bias_dbz, 6);
  assert.match(harness.element('#reference-description').textContent, /강도 \+6 dBZ/);
  await harness.click('#clear-reference');
  harness.element('#lead-minutes').value = '60';
  await harness.submit();
  assert.equal(harness.requests.at(-1).payload.reference, null);
  assert.equal(harness.element('#comparison-results').hidden, true);
});
