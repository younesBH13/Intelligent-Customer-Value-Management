import React, { useEffect, useMemo, useState } from 'react';
import {
  Activity,
  Bot,
  Brain,
  Cpu,
  Database,
  Download,
  GitBranch,
  Layers,
  ShieldAlert,
  Sparkles,
  Target,
  Users,
} from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { api } from './api';

const SEGMENT_COLORS = {
  'High-Value Loyal': '#34d399',
  'High-Value At Risk': '#f87171',
  Growing: '#38bdf8',
  Occasional: '#fbbf24',
  'Inactive / Declining': '#a78bfa',
};

const NAV_ITEMS = [
  { id: 'overview', label: 'Executive Overview', icon: Activity },
  { id: 'segmentation', label: 'Module 1: Segmentation', icon: Users },
  { id: 'models', label: 'Module 1–2: CLV & Churn', icon: Brain },
  { id: 'nba', label: 'Module 3–4: Offers & NBA', icon: Target },
  { id: 'xai', label: 'Explainable AI (SHAP)', icon: Sparkles },
  { id: 'uplift', label: 'Uplift Sim & NLP', icon: GitBranch },
  { id: 'twin', label: 'Digital Twin & AI Agent', icon: Bot },
];

function MetricCard({ label, value, sub }) {
  return (
    <div className="card">
      <div className="kpi-label">{label}</div>
      <div className="kpi-val">{value}</div>
      {sub && <div className="kpi-sub">{sub}</div>}
    </div>
  );
}

export default function App() {
  const [datasets, setDatasets] = useState([]);
  const [datasetKey, setDatasetKey] = useState('iranian');
  const [activeTab, setActiveTab] = useState('overview');
  const [report, setReport] = useState(null);
  const [samplePoints, setSamplePoints] = useState([]);
  const [customers, setCustomers] = useState([]);
  const [segFilter, setSegFilter] = useState('');
  const [riskFilter, setRiskFilter] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  // XAI Inspector state
  const [selectedCid, setSelectedCid] = useState('');
  const [xaiData, setXaiData] = useState(null);
  const [xaiLoading, setXaiLoading] = useState(false);

  // Digital Twin state
  const [twinCid, setTwinCid] = useState('');
  const [twinOverrides, setTwinOverrides] = useState({});
  const [twinResult, setTwinResult] = useState(null);

  // AI Agent state
  const [agentQuery, setAgentQuery] = useState('');
  const [agentRes, setAgentRes] = useState(null);
  const [agentLoading, setAgentLoading] = useState(false);

  useEffect(() => {
    api.getDatasets().then(setDatasets).catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    let cancelled = false;
    async function loadAll() {
      setLoading(true);
      setError('');
      try {
        const [rep, pts, custRes] = await Promise.all([
          api.getReport(datasetKey),
          api.getPortfolioSample(datasetKey, 400),
          api.getCustomers(datasetKey, { limit: 150 }),
        ]);
        if (cancelled) return;
        setReport(rep);
        setSamplePoints(pts);
        setCustomers(custRes.items || []);
        const firstId = custRes.items?.[0]?.[rep.dataset.id_col] || '';
        setSelectedCid(firstId);
        setTwinCid(firstId);
        setTwinOverrides(
          datasetKey === 'iranian'
            ? { complains: 0, call_failure: 0, charge_amount: 3, subscription_length: 24 }
            : { contract_type: 'One year', tenure_months: 24, monthly_charges: 65, tech_support_yes: 1 }
        );
        setTwinResult(null);
        setAgentRes(null);
      } catch (e) {
        if (!cancelled) setError(e.message);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    loadAll();
    return () => {
      cancelled = true;
    };
  }, [datasetKey]);

  useEffect(() => {
    if (!selectedCid || activeTab !== 'xai') return;
    setXaiLoading(true);
    api
      .explainCustomer(datasetKey, selectedCid)
      .then(setXaiData)
      .catch((e) => setError(e.message))
      .finally(() => setXaiLoading(false));
  }, [datasetKey, selectedCid, activeTab]);

  const filteredCustomers = useMemo(() => {
    return customers.filter((c) => {
      if (segFilter && c.segment_name !== segFilter) return false;
      if (riskFilter && c.risk_tier !== riskFilter) return false;
      return true;
    });
  }, [customers, segFilter, riskFilter]);

  const segmentGroups = useMemo(() => {
    const map = {};
    samplePoints.forEach((p) => {
      const seg = p.segment_name || 'Other';
      if (!map[seg]) map[seg] = [];
      map[seg].push(p);
    });
    return map;
  }, [samplePoints]);

  const handleRunTwin = async () => {
    try {
      const res = await api.simulateTwin(datasetKey, {
        base_customer_id: twinCid || null,
        overrides: twinOverrides,
      });
      setTwinResult(res);
    } catch (e) {
      setError(e.message);
    }
  };

  const handleAskAgent = async (qText) => {
    const queryToSend = qText ?? agentQuery;
    if (!queryToSend.trim()) return;
    setAgentQuery(queryToSend);
    setAgentLoading(true);
    try {
      const res = await api.askAgent(datasetKey, queryToSend);
      setAgentRes(res);
    } catch (e) {
      setError(e.message);
    } finally {
      setAgentLoading(false);
    }
  };

  const handleExportAggregateCsv = () => {
    if (!report) return;
    const rows = report.stage5_recommendations.offer_distribution || [];
    const headers = ['recommended_offer_id', 'recommended_offer', 'customer_count', 'share_pct', 'avg_churn_prob', 'avg_accept_prob', 'avg_predicted_clv', 'total_cost'];
    const csv = [
      headers.join(','),
      ...rows.map((r) => headers.map((h) => JSON.stringify(r[h] ?? '')).join(',')),
    ].join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${datasetKey}_aggregate_cvm_report.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-badge">CVM</div>
          <div>
            <div>Intelligent CVM</div>
            <div style={{ fontSize: 11, color: 'var(--muted)', fontWeight: 500 }}>
              React + FastAPI Platform
            </div>
          </div>
        </div>

        <div>
          <div className="kpi-label" style={{ marginBottom: 6 }}>Active Dataset</div>
          <select
            className="select"
            value={datasetKey}
            onChange={(e) => setDatasetKey(e.target.value)}
          >
            {datasets.map((d) => (
              <option key={d.key} value={d.key}>
                {d.name}
              </option>
            ))}
          </select>
        </div>

        <nav className="nav-list">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            return (
              <button
                key={item.id}
                className={`nav-btn ${activeTab === item.id ? 'active' : ''}`}
                onClick={() => setActiveTab(item.id)}
              >
                <Icon size={16} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        {report && (
          <div className="card" style={{ marginTop: 'auto', marginBottom: 0, padding: 12 }}>
            <div className="kpi-label">ML Engineering</div>
            <div style={{ fontSize: 12, marginTop: 6, color: 'var(--muted)' }}>
              <div><Database size={12} style={{ verticalAlign: 'middle' }} /> DB: <strong>{report.ml_engineering?.database?.backend || 'sqlite'}</strong></div>
              <div style={{ marginTop: 4 }}><Cpu size={12} style={{ verticalAlign: 'middle' }} /> MLflow Run: <code>{(report.ml_engineering?.mlflow_run_id || '').slice(0, 8)}</code></div>
            </div>
          </div>
        )}
      </aside>

      <main className="main">
        <div className="topbar">
          <div>
            <h1 className="page-title">
              {report ? report.dataset.name : 'Loading CVM Intelligence...'}
            </h1>
            <p className="page-sub">
              {report?.dataset.description || 'Connected 4-Module CVM Decision-Support System'}
            </p>
          </div>
          {report && (
            <button className="btn-secondary" onClick={handleExportAggregateCsv}>
              <Download size={14} style={{ verticalAlign: 'middle', marginRight: 6 }} />
              Export Aggregate CSV (No PII)
            </button>
          )}
        </div>

        {error && <div className="alert">Error: {error}</div>}
        {loading && <div className="card">Loading pipeline artifacts from FastAPI...</div>}

        {!loading && report && activeTab === 'overview' && (
          <>
            <div className="grid-4">
              <MetricCard
                label="Subscribers"
                value={report.stage1_data_quality.clean_rows.toLocaleString()}
                sub="Zero PII / Leakage Removed"
              />
              <MetricCard
                label="Observed Churn Rate"
                value={`${(report.stage1_data_quality.churn_rate * 100).toFixed(1)}%`}
                sub={`Best: ${report.stage4_churn_benchmark.best_model}`}
              />
              <MetricCard
                label={`Mean ${report.dataset.value_label}`}
                value={report.stage1_data_quality.value_summary.mean.toFixed(1)}
                sub={`Best CLV: ${report.stage3_clv_benchmark.best_model}`}
              />
              <MetricCard
                label="Value-Aware ROI (Sim.)"
                value={`${report.stage5_recommendations.policy_comparison.value_aware_cvm_policy.sim_roi_pct}%`}
                sub={`vs Generic: ${report.stage5_recommendations.policy_comparison.generic_churn_only_baseline.sim_roi_pct}%`}
              />
            </div>

            <div className="grid-2">
              <div className="card">
                <div className="card-title">Risk vs. Customer Value Portfolio (Sampled)</div>
                <ResponsiveContainer width="100%" height={320}>
                  <ScatterChart margin={{ top: 10, right: 16, bottom: 10, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#263248" />
                    <XAxis type="number" dataKey="predicted_clv" name="Predicted Value" stroke="#94a3b8" />
                    <YAxis type="number" dataKey="churn_prob" name="Churn Prob" stroke="#94a3b8" domain={[0, 1]} />
                    <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ background: '#111827', border: '1px solid #263248' }} />
                    <Legend />
                    {Object.entries(segmentGroups).map(([seg, pts]) => (
                      <Scatter
                        key={seg}
                        name={seg}
                        data={pts}
                        fill={SEGMENT_COLORS[seg.split(' (')[0]] || '#38bdf8'}
                      />
                    ))}
                  </ScatterChart>
                </ResponsiveContainer>
              </div>

              <div className="card">
                <div className="card-title">Module 4 — Policy Comparison (Top 30% Targeted)</div>
                <div className="alert">{report.stage5_recommendations.policy_comparison.note}</div>
                <table>
                  <thead>
                    <tr>
                      <th>Policy</th>
                      <th>Total Cost</th>
                      <th>Sim. Saved Value</th>
                      <th>Sim. Net Value</th>
                      <th>Sim. ROI</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>Generic Churn-Only Baseline</td>
                      <td>{report.stage5_recommendations.policy_comparison.generic_churn_only_baseline.total_cost.toLocaleString()}</td>
                      <td>{report.stage5_recommendations.policy_comparison.generic_churn_only_baseline.sim_saved_value.toLocaleString()}</td>
                      <td>{report.stage5_recommendations.policy_comparison.generic_churn_only_baseline.sim_net_value.toLocaleString()}</td>
                      <td><span className="badge badge-amber">{report.stage5_recommendations.policy_comparison.generic_churn_only_baseline.sim_roi_pct}%</span></td>
                    </tr>
                    <tr>
                      <td><strong>Value-Aware NBA Policy</strong></td>
                      <td>{report.stage5_recommendations.policy_comparison.value_aware_cvm_policy.total_cost.toLocaleString()}</td>
                      <td>{report.stage5_recommendations.policy_comparison.value_aware_cvm_policy.sim_saved_value.toLocaleString()}</td>
                      <td>{report.stage5_recommendations.policy_comparison.value_aware_cvm_policy.sim_net_value.toLocaleString()}</td>
                      <td><span className="badge badge-green">{report.stage5_recommendations.policy_comparison.value_aware_cvm_policy.sim_roi_pct}%</span></td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </>
        )}

        {!loading && report && activeTab === 'segmentation' && (
          <>
            <div className="grid-2">
              <div className="card">
                <div className="card-title">Clustering Benchmark (Selected: {report.stage2_segmentation.best_model})</div>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Algorithm</th>
                        <th>Clusters</th>
                        <th>Silhouette</th>
                        <th>Davies–Bouldin</th>
                        <th>Calinski–Harabasz</th>
                      </tr>
                    </thead>
                    <tbody>
                      {report.stage2_segmentation.metrics_table.map((m) => (
                        <tr key={m.algorithm}>
                          <td>{m.algorithm}</td>
                          <td>{m.n_clusters}</td>
                          <td>{m.silhouette ?? '—'}</td>
                          <td>{m.davies_bouldin ?? '—'}</td>
                          <td>{m.calinski_harabasz ?? '—'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              <div className="card">
                <div className="card-title">2D PCA Customer Segment Projection</div>
                <ResponsiveContainer width="100%" height={280}>
                  <ScatterChart>
                    <CartesianGrid strokeDasharray="3 3" stroke="#263248" />
                    <XAxis type="number" dataKey="pca_1" name="PCA 1" stroke="#94a3b8" />
                    <YAxis type="number" dataKey="pca_2" name="PCA 2" stroke="#94a3b8" />
                    <Tooltip contentStyle={{ background: '#111827', border: '1px solid #263248' }} />
                    <Legend />
                    {Object.entries(segmentGroups).map(([seg, pts]) => (
                      <Scatter
                        key={seg}
                        name={seg}
                        data={pts}
                        fill={SEGMENT_COLORS[seg.split(' (')[0]] || '#38bdf8'}
                      />
                    ))}
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="card">
              <div className="card-title">Segment Profiles</div>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Segment</th>
                      <th>Customers</th>
                      <th>Share %</th>
                      <th>Avg Value</th>
                      <th>Churn Rate</th>
                    </tr>
                  </thead>
                  <tbody>
                    {report.stage2_segmentation.cluster_profiles.map((p) => (
                      <tr key={p.segment_name}>
                        <td><span className="badge badge-blue">{p.segment_name}</span></td>
                        <td>{p.customer_count.toLocaleString()}</td>
                        <td>{p.share_pct}%</td>
                        <td>{p.avg_value.toLocaleString()}</td>
                        <td>{(p.churn_rate * 100).toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        )}

        {!loading && report && activeTab === 'models' && (
          <>
            <div className="grid-2">
              <div className="card">
                <div className="card-title">CLV / Value Benchmark (Best: {report.stage3_clv_benchmark.best_model})</div>
                <table>
                  <thead>
                    <tr><th>Model</th><th>MAE</th><th>RMSE</th><th>R²</th><th>MAPE %</th></tr>
                  </thead>
                  <tbody>
                    {report.stage3_clv_benchmark.metrics_table.map((m) => (
                      <tr key={m.model}>
                        <td>{m.model}</td>
                        <td>{m.mae}</td>
                        <td>{m.rmse}</td>
                        <td>{m.r2}</td>
                        <td>{m.mape_pct ?? '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="card">
                <div className="card-title">Churn Benchmark — 5-Fold CV & Calibration (Best: {report.stage4_churn_benchmark.best_model})</div>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr><th>Model</th><th>PR-AUC</th><th>5-Fold PR-AUC</th><th>ROC-AUC</th><th>Recall</th><th>ECE</th></tr>
                    </thead>
                    <tbody>
                      {report.stage4_churn_benchmark.metrics_table.map((m) => (
                        <tr key={m.model}>
                          <td>{m.model}</td>
                          <td>{m.pr_auc}</td>
                          <td>{m.cv5_pr_auc}</td>
                          <td>{m.roc_auc}</td>
                          <td>{m.recall}</td>
                          <td>{m.ece}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            <div className="grid-2">
              <div className="card">
                <div className="card-title">Top Global SHAP Features — CLV Model</div>
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={report.stage6_explainability.clv_global_importance.slice(0, 8)} layout="vertical" margin={{ left: 40 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#263248" />
                    <XAxis type="number" stroke="#94a3b8" />
                    <YAxis type="category" dataKey="feature" width={130} stroke="#94a3b8" tick={{ fontSize: 11 }} />
                    <Tooltip contentStyle={{ background: '#111827', border: '1px solid #263248' }} />
                    <Bar dataKey="mean_abs_shap" fill="#38bdf8" />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="card">
                <div className="card-title">Top Global SHAP Features — Churn Model</div>
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={report.stage6_explainability.churn_global_importance.slice(0, 8)} layout="vertical" margin={{ left: 40 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#263248" />
                    <XAxis type="number" stroke="#94a3b8" />
                    <YAxis type="category" dataKey="feature" width={130} stroke="#94a3b8" tick={{ fontSize: 11 }} />
                    <Tooltip contentStyle={{ background: '#111827', border: '1px solid #263248' }} />
                    <Bar dataKey="mean_abs_shap" fill="#f87171" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </>
        )}

        {!loading && report && activeTab === 'nba' && (
          <>
            {report.stage5_recommendations.offer_acceptance_model && (
              <div className="alert-info">
                <strong>Module 3 — {report.stage5_recommendations.offer_acceptance_model.model}:</strong>{' '}
                ROC-AUC <code>{report.stage5_recommendations.offer_acceptance_model.roc_auc}</code> |{' '}
                PR-AUC <code>{report.stage5_recommendations.offer_acceptance_model.pr_auc}</code> |{' '}
                F1 <code>{report.stage5_recommendations.offer_acceptance_model.f1}</code>{' '}
                — {report.stage5_recommendations.offer_acceptance_model.note}
              </div>
            )}

            <div className="card">
              <div className="card-title">
                <span>Module 4 — Utility-Based Next Best Action Priority Queue</span>
                <div style={{ display: 'flex', gap: 10 }}>
                  <select className="select" style={{ width: 180 }} value={segFilter} onChange={(e) => setSegFilter(e.target.value)}>
                    <option value="">All Segments</option>
                    {report.stage2_segmentation.cluster_profiles.map((p) => (
                      <option key={p.segment_name} value={p.segment_name}>{p.segment_name}</option>
                    ))}
                  </select>
                  <select className="select" style={{ width: 150 }} value={riskFilter} onChange={(e) => setRiskFilter(e.target.value)}>
                    <option value="">All Risk Tiers</option>
                    <option value="High Risk">High Risk</option>
                    <option value="Medium Risk">Medium Risk</option>
                    <option value="Low Risk">Low Risk</option>
                  </select>
                </div>
              </div>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Rank</th>
                      <th>Customer ID</th>
                      <th>Segment</th>
                      <th>P(Churn)</th>
                      <th>Pred. Value</th>
                      <th>Priority</th>
                      <th>Next Best Offer</th>
                      <th>P(Accept)</th>
                      <th>Exp. Utility</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredCustomers.slice(0, 80).map((c) => {
                      const cid = c[report.dataset.id_col];
                      return (
                        <tr key={cid}>
                          <td>#{c.priority_rank}</td>
                          <td><code>{cid}</code></td>
                          <td>{c.segment_name}</td>
                          <td>{(c.churn_prob * 100).toFixed(1)}%</td>
                          <td>{Number(c.predicted_clv).toLocaleString()}</td>
                          <td><strong>{Number(c.priority_score).toLocaleString()}</strong></td>
                          <td><span className="badge badge-blue">{c.recommended_offer}</span></td>
                          <td>{c.offer_accept_prob != null ? `${(c.offer_accept_prob * 100).toFixed(1)}%` : '—'}</td>
                          <td>{Number(c.sim_expected_net_benefit).toLocaleString()}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        )}

        {!loading && report && activeTab === 'xai' && (
          <>
            <div className="card">
              <div className="card-title">
                <span>Individual Customer SHAP Inspector</span>
                <select
                  className="select"
                  style={{ width: 240 }}
                  value={selectedCid}
                  onChange={(e) => setSelectedCid(e.target.value)}
                >
                  {customers.slice(0, 100).map((c) => {
                    const cid = c[report.dataset.id_col];
                    return <option key={cid} value={cid}>{cid} (Priority #{c.priority_rank})</option>;
                  })}
                </select>
              </div>
              <div className="alert-info">{report.stage6_explainability.note}</div>
              {xaiLoading && <div>Computing SHAP feature attributions...</div>}
              {!xaiLoading && xaiData && (
                <>
                  <div className="grid-4">
                    <MetricCard label="Segment" value={xaiData.segment_name} />
                    <MetricCard label="Churn Risk" value={`${(xaiData.churn_prob * 100).toFixed(1)}%`} />
                    <MetricCard label="Predicted Value" value={xaiData.predicted_clv.toLocaleString()} />
                    <MetricCard label="Recommended Offer" value={xaiData.recommended_offer} />
                  </div>
                  <div className="grid-2">
                    <div className="card" style={{ marginBottom: 0 }}>
                      <div className="card-title">Local SHAP — Churn Risk Drivers</div>
                      <ResponsiveContainer width="100%" height={280}>
                        <BarChart data={[...xaiData.churn_shap_contributions].reverse()} layout="vertical" margin={{ left: 30 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#263248" />
                          <XAxis type="number" stroke="#94a3b8" />
                          <YAxis type="category" dataKey="feature" width={130} stroke="#94a3b8" tick={{ fontSize: 11 }} />
                          <Tooltip contentStyle={{ background: '#111827', border: '1px solid #263248' }} />
                          <Bar dataKey="shap_contribution">
                            {[...xaiData.churn_shap_contributions].reverse().map((entry, idx) => (
                              <Cell key={idx} fill={entry.shap_contribution >= 0 ? '#f87171' : '#34d399'} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                    <div className="card" style={{ marginBottom: 0 }}>
                      <div className="card-title">Local SHAP — CLV / Value Drivers</div>
                      <ResponsiveContainer width="100%" height={280}>
                        <BarChart data={[...xaiData.clv_shap_contributions].reverse()} layout="vertical" margin={{ left: 30 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#263248" />
                          <XAxis type="number" stroke="#94a3b8" />
                          <YAxis type="category" dataKey="feature" width={130} stroke="#94a3b8" tick={{ fontSize: 11 }} />
                          <Tooltip contentStyle={{ background: '#111827', border: '1px solid #263248' }} />
                          <Bar dataKey="shap_contribution">
                            {[...xaiData.clv_shap_contributions].reverse().map((entry, idx) => (
                              <Cell key={idx} fill={entry.shap_contribution >= 0 ? '#38bdf8' : '#fbbf24'} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </>
              )}
            </div>
          </>
        )}

        {!loading && report && activeTab === 'uplift' && (
          <div className="grid-2">
            <div className="card">
              <div className="card-title">Semi-Synthetic Uplift Modeling (S / T / X-Learner)</div>
              <div className="alert">{report.stage5b_uplift_simulation?.note}</div>
              <table>
                <thead><tr><th>Learner</th><th>AUUC</th><th>Uplift @ 30%</th><th>PEHE RMSE</th></tr></thead>
                <tbody>
                  {(report.stage5b_uplift_simulation?.metrics_table || []).map((u) => (
                    <tr key={u.learner}>
                      <td>{u.learner}</td>
                      <td>{u.auuc}</td>
                      <td>{u.uplift_at_30pct}</td>
                      <td>{u.pehe_rmse}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="card">
              <div className="card-title">Optional NLP — Dissatisfaction Themes</div>
              {report.optional_nlp_insights?.available ? (
                <>
                  <div className="alert-info">{report.optional_nlp_insights.note}</div>
                  <table>
                    <thead><tr><th>Topic</th><th>Keywords</th><th>Count</th><th>Share</th></tr></thead>
                    <tbody>
                      {report.optional_nlp_insights.topics.map((t) => (
                        <tr key={t.topic_id}>
                          <td>Topic {t.topic_id}</td>
                          <td>{t.theme_keywords}</td>
                          <td>{t.document_count}</td>
                          <td>{t.share_pct}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </>
              ) : (
                <div className="alert-info">{report.optional_nlp_insights?.note}</div>
              )}
            </div>
          </div>
        )}

        {!loading && report && activeTab === 'twin' && (
          <div className="grid-2">
            <div className="card">
              <div className="card-title">Customer Digital Twin & Live Prediction</div>
              <div style={{ marginBottom: 12 }}>
                <div className="kpi-label" style={{ marginBottom: 4 }}>Select Base Customer</div>
                <select className="select" value={twinCid} onChange={(e) => setTwinCid(e.target.value)}>
                  <option value="">New Customer (Median Baseline)</option>
                  {customers.slice(0, 80).map((c) => {
                    const cid = c[report.dataset.id_col];
                    return <option key={cid} value={cid}>{cid} (Priority #{c.priority_rank})</option>;
                  })}
                </select>
              </div>

              {datasetKey === 'iranian' ? (
                <div className="grid-2">
                  <div>
                    <div className="kpi-label">Complaints (0 or 1)</div>
                    <input className="input" type="number" min={0} max={1} value={twinOverrides.complains ?? 0} onChange={(e) => setTwinOverrides({ ...twinOverrides, complains: Number(e.target.value) })} />
                  </div>
                  <div>
                    <div className="kpi-label">Call Failures</div>
                    <input className="input" type="number" min={0} max={40} value={twinOverrides.call_failure ?? 0} onChange={(e) => setTwinOverrides({ ...twinOverrides, call_failure: Number(e.target.value) })} />
                  </div>
                  <div>
                    <div className="kpi-label">Charge Level (0–10)</div>
                    <input className="input" type="number" min={0} max={10} value={twinOverrides.charge_amount ?? 2} onChange={(e) => setTwinOverrides({ ...twinOverrides, charge_amount: Number(e.target.value) })} />
                  </div>
                  <div>
                    <div className="kpi-label">Subscription Length (mo)</div>
                    <input className="input" type="number" min={1} max={60} value={twinOverrides.subscription_length ?? 24} onChange={(e) => setTwinOverrides({ ...twinOverrides, subscription_length: Number(e.target.value) })} />
                  </div>
                </div>
              ) : (
                <div className="grid-2">
                  <div>
                    <div className="kpi-label">Contract Type</div>
                    <select className="select" value={twinOverrides.contract_type || 'One year'} onChange={(e) => setTwinOverrides({ ...twinOverrides, contract_type: e.target.value })}>
                      <option value="Month-to-month">Month-to-month</option>
                      <option value="One year">One year</option>
                      <option value="Two year">Two year</option>
                    </select>
                  </div>
                  <div>
                    <div className="kpi-label">Tenure (Months)</div>
                    <input className="input" type="number" min={0} max={72} value={twinOverrides.tenure_months ?? 24} onChange={(e) => setTwinOverrides({ ...twinOverrides, tenure_months: Number(e.target.value) })} />
                  </div>
                  <div>
                    <div className="kpi-label">Monthly Charges ($)</div>
                    <input className="input" type="number" min={18} max={120} value={twinOverrides.monthly_charges ?? 65} onChange={(e) => setTwinOverrides({ ...twinOverrides, monthly_charges: Number(e.target.value) })} />
                  </div>
                  <div>
                    <div className="kpi-label">Tech Support (0/1)</div>
                    <input className="input" type="number" min={0} max={1} value={twinOverrides.tech_support_yes ?? 1} onChange={(e) => setTwinOverrides({ ...twinOverrides, tech_support_yes: Number(e.target.value) })} />
                  </div>
                </div>
              )}

              <button className="btn" style={{ marginTop: 12 }} onClick={handleRunTwin}>
                Run Digital Twin Simulation
              </button>

              {twinResult && (
                <div className="card" style={{ marginTop: 14, background: 'var(--surface-2)' }}>
                  <div className="card-title">Simulation Output</div>
                  <div>Churn Risk: <strong>{(twinResult.simulated.churn_prob * 100).toFixed(1)}%</strong> {twinResult.deltas && <span>({(twinResult.deltas.delta_churn_prob * 100).toFixed(1)}%)</span>}</div>
                  <div>Predicted Value: <strong>{twinResult.simulated.predicted_clv.toLocaleString()}</strong></div>
                  <div>Priority Score: <strong>{twinResult.simulated.priority_score.toLocaleString()}</strong></div>
                  <div style={{ marginTop: 6 }}>Recommended Offer: <span className="badge badge-green">{twinResult.simulated.recommended_offer}</span></div>
                </div>
              )}
            </div>

            <div className="card">
              <div className="card-title">CVM AI Agent (Retention Copilot)</div>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 12 }}>
                <button className="btn-secondary" onClick={() => handleAskAgent('Show top 5 priority customers')}>Top 5 Priority</button>
                <button className="btn-secondary" onClick={() => handleAskAgent('Compare segments')}>Compare Segments</button>
                {customers[0] && (
                  <>
                    <button className="btn-secondary" onClick={() => handleAskAgent(`Explain customer ${customers[0][report.dataset.id_col]}`)}>
                      Explain {customers[0][report.dataset.id_col]}
                    </button>
                    <button className="btn-secondary" onClick={() => handleAskAgent(`Simulate digital twin for ${customers[0][report.dataset.id_col]}`)}>
                      Twin {customers[0][report.dataset.id_col]}
                    </button>
                  </>
                )}
              </div>

              <div style={{ display: 'flex', gap: 8 }}>
                <input
                  className="input"
                  placeholder="Ask about top priority customers, segments, SHAP, or Digital Twin..."
                  value={agentQuery}
                  onChange={(e) => setAgentQuery(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleAskAgent()}
                />
                <button className="btn" onClick={() => handleAskAgent()}>Ask</button>
              </div>

              {agentLoading && <div style={{ marginTop: 12 }}>Thinking...</div>}
              {agentRes && (
                <div className="card" style={{ marginTop: 14, background: 'var(--surface-2)', whiteSpace: 'pre-wrap', fontSize: 13.5, lineHeight: 1.55 }}>
                  {agentRes.answer}
                </div>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

