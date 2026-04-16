/* ── Slider fill on load ──────────────────────────────────── */
function updateSliderFill(el) {
  const pct = ((el.value - el.min) / (el.max - el.min)) * 100;
  el.style.setProperty('--fill', pct + '%');
}
document.querySelectorAll('.slider').forEach(s => updateSliderFill(s));

/* ── Toggle helpers ───────────────────────────────────────── */
function setToggle(groupId, val) {
  const group = document.getElementById(groupId);
  group.querySelectorAll('.toggle-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.val === val);
  });
}
function getToggle(groupId) {
  const active = document.querySelector(`#${groupId} .toggle-btn.active`);
  return active ? active.dataset.val : 'No';
}

/* ── Service toggles ──────────────────────────────────────── */
function toggleService(id) {
  const el = document.getElementById(id);
  el.classList.toggle('active');
  updateServiceDots();
}
function updateServiceDots() {
  const svcs = document.querySelectorAll('.service-toggle');
  const dots = document.querySelectorAll('#svc-dots .dot');
  let count = 0;
  svcs.forEach(s => { if (s.classList.contains('active')) count++; });
  dots.forEach((d, i) => d.classList.toggle('active', i < count));
}
updateServiceDots();

/* ── Billing auto-calc ────────────────────────────────────── */
function updateCalcs() {
  const m = parseFloat(document.getElementById('monthly').value) || 0;
  const t = parseFloat(document.getElementById('total').value) || 0;
  const ten = parseInt(document.getElementById('tenure').value) || 0;
  const avg = t / (ten + 1);
  document.getElementById('avg-spend').textContent = '$' + avg.toFixed(2);

  // Segment bar
  const pct = Math.min(100, (m / 120) * 100);
  document.getElementById('seg-fill').style.width = pct + '%';
  let seg = 'Budget';
  if (m > 80) seg = 'Premium';
  else if (m > 45) seg = 'Mid-tier';
  document.getElementById('seg-label').textContent = seg;
}
document.getElementById('monthly').addEventListener('input', updateCalcs);
document.getElementById('total').addEventListener('input', updateCalcs);
document.getElementById('tenure').addEventListener('input', updateCalcs);
updateCalcs();

/* ── Gauge animation ──────────────────────────────────────── */
function animateGauge(prob) {
  const path = document.getElementById('gauge-fill');
  const needle = document.getElementById('gauge-needle');
  const total = 251.2;

  // prob 0→1 maps to offset total→0
  const targetOffset = total * (1 - prob);
  const targetAngle = -90 + prob * 180; // -90° (left) to +90° (right)

  // Animate over ~800ms
  let start = null;
  const initOffset = parseFloat(path.getAttribute('stroke-dashoffset'));
  function step(ts) {
    if (!start) start = ts;
    const progress = Math.min((ts - start) / 800, 1);
    const ease = 1 - Math.pow(1 - progress, 3);
    path.setAttribute('stroke-dashoffset', initOffset + (targetOffset - initOffset) * ease);
    needle.style.transform = `rotate(${targetAngle * ease - 90 * (1 - ease)}deg)`;
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

/* ── Main prediction function ─────────────────────────────── */
async function submitPrediction() {
  const btn = document.getElementById('predict-btn');
  btn.classList.add('loading');
  btn.disabled = true;

  // Gather form data
  const payload = {
    tenure: parseInt(document.getElementById('tenure').value),
    monthly_charges: parseFloat(document.getElementById('monthly').value),
    total_charges: parseFloat(document.getElementById('total').value),
    contract: document.getElementById('contract').value,
    payment: document.getElementById('payment').value,
    internet_service: document.getElementById('internet').value,
    online_security: document.getElementById('svc-security').classList.contains('active') ? 'Yes' : 'No',
    online_backup: document.getElementById('svc-backup').classList.contains('active') ? 'Yes' : 'No',
    device_protection: document.getElementById('svc-device').classList.contains('active') ? 'Yes' : 'No',
    tech_support: document.getElementById('svc-tech').classList.contains('active') ? 'Yes' : 'No',
    streaming_tv: document.getElementById('svc-tv').classList.contains('active') ? 'Yes' : 'No',
    streaming_movies: document.getElementById('svc-movies').classList.contains('active') ? 'Yes' : 'No',
    paperless: getToggle('paper-group'),
    senior_citizen: getToggle('senior-group')
  };

  let prob, pred;

  try {
    const resp = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!resp.ok) throw new Error('Server error');
    const data = await resp.json();
    prob = data.probability;
    pred = data.prediction;
  } catch (e) {
    // Demo fallback if backend not running
    prob = computeDemoProb(payload);
    pred = prob > 0.45 ? 1 : 0;
  }

  btn.classList.remove('loading');
  btn.disabled = false;

  showResult(prob, pred, payload);
}

/* ── Demo probability (fallback if no Flask backend) ─────── */
function computeDemoProb(p) {
  let s = 0.12;
  if (p.contract === 'Month-to-month') s += 0.28;
  else if (p.contract === 'One year') s += 0.08;
  if (p.internet_service === 'Fiber optic') s += 0.15;
  if (p.tenure <= 12) s += 0.18;
  if (p.monthly_charges > 80) s += 0.10;
  if (p.senior_citizen === 'Yes') s += 0.06;
  if (p.online_security === 'No') s += 0.05;
  if (p.tech_support === 'No') s += 0.04;
  return Math.min(0.97, Math.max(0.04, s));
}

/* ── Render results ───────────────────────────────────────── */
function showResult(prob, pred, payload) {
  const panel = document.getElementById('result-panel');
  panel.style.display = 'block';

  const pct = Math.round(prob * 100);
  const isHigh = prob >= 0.45;

  // Gauge
  animateGauge(prob);
  document.getElementById('gauge-pct').textContent = pct + '%';

  // Verdict
  const badge = document.getElementById('verdict-badge');
  badge.className = 'verdict-badge ' + (isHigh ? 'high' : 'low');
  document.getElementById('verdict-icon').textContent = isHigh ? '⚠️' : '✅';
  document.getElementById('verdict-title').textContent = isHigh ? 'High Churn Risk' : 'Low Churn Risk';
  document.getElementById('verdict-sub').textContent = isHigh
    ? 'This customer is likely to churn. Act now.'
    : 'This customer is likely to stay. Consider upselling.';

  // Metric pills
  let riskLevel = prob < 0.25 ? 'Low' : prob < 0.55 ? 'Moderate' : 'Critical';
  let keyDriver = 'Contract type';
  if (payload.tenure <= 12) keyDriver = 'New customer';
  else if (payload.internet_service === 'Fiber optic') keyDriver = 'Fiber optic';
  else if (payload.monthly_charges > 80) keyDriver = 'High charges';

  document.getElementById('risk-level').textContent = riskLevel;
  document.getElementById('key-driver').textContent = keyDriver;
  document.getElementById('confidence').textContent = (75 + Math.round(Math.abs(prob - 0.5) * 40)) + '%';

  // Risk marker
  document.getElementById('risk-marker').style.left = Math.min(95, Math.max(2, pct - 2)) + '%';

  // Actions
  const actions = buildActions(prob, payload);
  const list = document.getElementById('actions-list');
  list.innerHTML = actions.map((a, i) => `
    <div class="action-card priority-${a.priority}" style="animation-delay:${i * .08}s">
      <div class="action-card-top">
        <span class="action-title">${a.title}</span>
        <span class="action-badge">${a.priority}</span>
      </div>
      <p class="action-desc">${a.desc}</p>
    </div>
  `).join('');

  panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/* ── Build actions list ───────────────────────────────────── */
function buildActions(prob, p) {
  const actions = [];
  if (prob < 0.25) {
    actions.push({ priority: 'low', title: 'Loyalty Reward', desc: 'Customer is happy — offer a referral bonus or premium add-on at a discounted rate.' });
    actions.push({ priority: 'low', title: 'Upsell Opportunity', desc: 'Low churn risk is the ideal window to introduce higher-tier plans.' });
    return actions;
  }
  if (p.contract === 'Month-to-month')
    actions.push({ priority: 'high', title: 'Upgrade Contract', desc: 'Offer 20–25% discount to lock in a one-year or two-year contract immediately.' });
  if (p.tenure <= 12)
    actions.push({ priority: 'high', title: 'Onboarding Support', desc: 'Assign a dedicated customer success manager — early-stage churn is preventable with personal attention.' });
  if (p.monthly_charges > 80)
    actions.push({ priority: 'medium', title: 'Pricing Review', desc: 'High monthly charges are a top churn driver. Offer a temporary 15% discount or bundle rebate.' });
  if (p.internet_service === 'Fiber optic')
    actions.push({ priority: 'medium', title: 'Service Quality Check', desc: 'Fiber optic customers churn more. Proactively confirm speed satisfaction and uptime.' });
  if (p.online_security === 'No')
    actions.push({ priority: 'medium', title: 'Security Add-on', desc: 'Offer free 3-month trial of Online Security — it increases perceived value and stickiness.' });
  if (p.tech_support === 'No')
    actions.push({ priority: 'medium', title: 'Tech Support Trial', desc: 'Customers without tech support churn significantly more. Offer a complementary onboarding call.' });
  if (p.senior_citizen === 'Yes')
    actions.push({ priority: 'low', title: 'Senior Assistance Plan', desc: 'Consider a dedicated senior helpline or simplified billing — increases satisfaction in this segment.' });
  if (actions.length === 0)
    actions.push({ priority: 'medium', title: 'Proactive Outreach', desc: 'Schedule a check-in call to understand satisfaction and preempt unvoiced concerns.' });
  return actions;
}