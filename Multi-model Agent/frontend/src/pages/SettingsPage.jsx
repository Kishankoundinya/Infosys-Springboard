import React, { useState, useEffect } from 'react';
import { Settings, Save, CheckCircle2, AlertCircle, Cpu, Sliders } from 'lucide-react';
import { api } from '../services/api';

export default function SettingsPage() {
  const [config, setConfig] = useState(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    try {
      const data = await api.getSettings();
      setConfig(data);
    } catch (err) {
      setError(err.message || 'Failed to load settings.');
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async (e) => {
    e.preventDefault();
    setSaving(true);
    setMessage('');
    setError('');

    try {
      const updated = await api.updateSettings({
        temperature: parseFloat(config.temperature),
        max_new_tokens: parseInt(config.max_new_tokens, 10),
        chunk_size: parseInt(config.chunk_size, 10),
        chunk_overlap: parseInt(config.chunk_overlap, 10),
        top_k: parseInt(config.top_k, 10),
        max_retrieval_steps: parseInt(config.max_retrieval_steps, 10),
      });
      setConfig(updated);
      setMessage('Settings updated successfully.');
    } catch (err) {
      setError(err.message || 'Failed to save settings.');
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="p-8 text-center text-slate-400 text-xs">
        Loading system configuration...
      </div>
    );
  }

  return (
    <div className="max-w-3xl space-y-6">
      <div className="bg-white p-6 rounded-lg border border-slate-200 shadow-sm">
        <div className="flex items-center gap-2 mb-1">
          <Sliders className="w-5 h-5 text-blue-600" />
          <h3 className="text-base font-semibold text-slate-900">System & Model Configuration</h3>
        </div>
        <p className="text-xs text-slate-500 mb-6">
          Fine-tune the local shared TinyLlama LLM parameters and LangChain RAG chunking values. Defaults are optimized for laptop CPU/RAM efficiency.
        </p>

        {message && (
          <div className="mb-4 p-3 bg-emerald-50 border border-emerald-200 rounded text-emerald-700 text-xs flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 flex-shrink-0" />
            <span>{message}</span>
          </div>
        )}

        {error && (
          <div className="mb-4 p-3 bg-rose-50 border border-rose-200 rounded text-rose-700 text-xs flex items-center gap-2">
            <AlertCircle className="w-4 h-4 flex-shrink-0" />
            <span>{error}</span>
          </div>
        )}

        <form onSubmit={handleSave} className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1">LLM Model (Configured)</label>
              <input
                type="text"
                value={config?.llm_model || ''}
                disabled
                className="w-full text-xs border border-slate-200 bg-slate-50 text-slate-500 rounded px-3 py-2 cursor-not-allowed font-mono"
              />
              <span className="text-[10px] text-slate-400">Configurable via .env (LLM_MODEL)</span>
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1">Inference Device</label>
              <input
                type="text"
                value={config?.device || ''}
                disabled
                className="w-full text-xs border border-slate-200 bg-slate-50 text-slate-500 rounded px-3 py-2 cursor-not-allowed font-mono uppercase"
              />
              <span className="text-[10px] text-slate-400">Configurable via .env (auto / cpu / mps)</span>
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1">Embedding Model</label>
              <input
                type="text"
                value={config?.embedding_model || ''}
                disabled
                className="w-full text-xs border border-slate-200 bg-slate-50 text-slate-500 rounded px-3 py-2 cursor-not-allowed font-mono"
              />
              <span className="text-[10px] text-slate-400">Lightweight Sentence Transformers</span>
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1">Temperature (Deterministic)</label>
              <input
                type="number"
                step="0.05"
                min="0.0"
                max="1.0"
                value={config?.temperature || 0.2}
                onChange={(e) => setConfig({ ...config, temperature: e.target.value })}
                className="w-full text-xs border border-slate-300 rounded px-3 py-2 text-slate-800 focus:outline-none focus:ring-1 focus:ring-blue-500"
              />
              <span className="text-[10px] text-slate-400">Default: 0.2 for deterministic financial answers</span>
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1">Max New Tokens (Research)</label>
              <input
                type="number"
                step="32"
                min="64"
                max="1024"
                value={config?.max_new_tokens || 256}
                onChange={(e) => setConfig({ ...config, max_new_tokens: e.target.value })}
                className="w-full text-xs border border-slate-300 rounded px-3 py-2 text-slate-800 focus:outline-none focus:ring-1 focus:ring-blue-500"
              />
              <span className="text-[10px] text-slate-400">Optimized for laptop CPU (256 default)</span>
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1">Top K Retrieved Chunks</label>
              <input
                type="number"
                min="1"
                max="15"
                value={config?.top_k || 5}
                onChange={(e) => setConfig({ ...config, top_k: e.target.value })}
                className="w-full text-xs border border-slate-300 rounded px-3 py-2 text-slate-800 focus:outline-none focus:ring-1 focus:ring-blue-500"
              />
              <span className="text-[10px] text-slate-400">Default: 5 chunks passed to LLM context</span>
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1">Chunk Size (Characters)</label>
              <input
                type="number"
                step="50"
                min="200"
                max="4000"
                value={config?.chunk_size || 1000}
                onChange={(e) => setConfig({ ...config, chunk_size: e.target.value })}
                className="w-full text-xs border border-slate-300 rounded px-3 py-2 text-slate-800 focus:outline-none focus:ring-1 focus:ring-blue-500"
              />
              <span className="text-[10px] text-slate-400">Default: 1000 characters per chunk</span>
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1">Chunk Overlap</label>
              <input
                type="number"
                step="10"
                min="0"
                max="500"
                value={config?.chunk_overlap || 150}
                onChange={(e) => setConfig({ ...config, chunk_overlap: e.target.value })}
                className="w-full text-xs border border-slate-300 rounded px-3 py-2 text-slate-800 focus:outline-none focus:ring-1 focus:ring-blue-500"
              />
              <span className="text-[10px] text-slate-400">Default: 150 characters overlap</span>
            </div>
          </div>

          <div className="pt-4 border-t border-slate-200 flex justify-end">
            <button
              type="submit"
              disabled={saving}
              className="px-5 py-2.5 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-300 text-white rounded text-xs font-semibold flex items-center gap-1.5 shadow-sm transition-colors"
            >
              <Save className="w-4 h-4" />
              <span>{saving ? 'Saving...' : 'Save Settings'}</span>
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
