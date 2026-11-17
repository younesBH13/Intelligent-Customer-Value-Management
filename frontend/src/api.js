const API_BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  });
  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`API ${res.status}: ${errText}`);
  }
  return res.json();
}

export const api = {
  getDatasets: () => request('/datasets'),
  getReport: (key) => request(`/report/${key}`),
  getPortfolioSample: (key, n = 450) => request(`/portfolio-sample/${key}?n=${n}`),
  getCustomers: (key, params = {}) => {
    const qs = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined && v !== null && v !== '') qs.set(k, v);
    });
    return request(`/customers/${key}?${qs.toString()}`);
  },
  explainCustomer: (key, customerId) => request(`/explain/${key}/${encodeURIComponent(customerId)}`),
  simulateTwin: (key, payload) =>
    request(`/predict/${key}`, { method: 'POST', body: JSON.stringify(payload) }),
  askAgent: (key, query) =>
    request(`/agent/${key}/ask`, { method: 'POST', body: JSON.stringify({ query }) }),
};

