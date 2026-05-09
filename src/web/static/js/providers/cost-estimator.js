/**
 * Cost Estimator - estimate translation cost in USD before launching.
 *
 * Triggers on model change, file added/removed, and (debounced) text input.
 * Pricing for OpenRouter/Poe is read from the model option's pricing data
 * already returned by their APIs. For other paid providers, defaults come
 * from the backend; users can override per model via the Edit Prices modal
 * (overrides persist in localStorage).
 */

import { ApiClient } from '../core/api-client.js';
import { DomHelpers } from '../ui/dom-helpers.js';
import { StateManager } from '../core/state-manager.js';

const STORAGE_KEY = 'tbl_pricing_overrides_v1';
const DEBOUNCE_MS = 800;

const LOCAL_PROVIDERS = new Set(['ollama']);
const API_PRICING_PROVIDERS = new Set(['openrouter', 'poe']);

let pricingDefaults = null;
let pricingLastUpdated = null;
let debounceTimer = null;
let inFlightController = null;
let listenersAttached = false;

function loadOverrides() {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) return {};
        const parsed = JSON.parse(raw);
        return (parsed && typeof parsed === 'object') ? parsed : {};
    } catch {
        return {};
    }
}

function saveOverrides(overrides) {
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(overrides));
    } catch {
        /* ignore quota errors */
    }
}

function getOverride(provider, model) {
    const all = loadOverrides();
    return all?.[provider]?.[model] || null;
}

function setOverride(provider, model, pricing) {
    const all = loadOverrides();
    if (!all[provider]) all[provider] = {};
    all[provider][model] = pricing;
    saveOverrides(all);
}

function clearOverride(provider, model) {
    const all = loadOverrides();
    if (all?.[provider]?.[model]) {
        delete all[provider][model];
        saveOverrides(all);
    }
}

function getCurrentProvider() {
    return DomHelpers.getValue('llmProvider');
}

function getCurrentModel() {
    return DomHelpers.getValue('model');
}

function getInputText() {
    const el = DomHelpers.getElement('inputText');
    return el ? (el.value || '') : '';
}

function getCurrentFilePath() {
    const files = StateManager.getState('files.toProcess') || [];
    if (!Array.isArray(files) || files.length === 0) return null;
    const first = files.find(f => f?.filePath) || files[0];
    return first?.filePath || null;
}

function getLanguagePair() {
    const src = DomHelpers.getValue('sourceLang') || '';
    const tgt = DomHelpers.getValue('targetLang') || '';
    return { src, tgt };
}

function getOptions() {
    return {
        refine: !!DomHelpers.getElement('refineTranslation')?.checked,
        text_cleanup: !!DomHelpers.getElement('textCleanup')?.checked,
    };
}

function readPricingFromModelOption() {
    const select = DomHelpers.getElement('model');
    if (!select) return null;
    const opt = select.selectedOptions?.[0];
    if (!opt) return null;
    const inAttr = opt.getAttribute('data-pricing-input');
    const outAttr = opt.getAttribute('data-pricing-output');
    if (inAttr === null || outAttr === null) return null;
    const input = parseFloat(inAttr);
    const output = parseFloat(outAttr);
    if (!Number.isFinite(input) || !Number.isFinite(output)) return null;
    return { input, output };
}

function resolvePricing(provider, model) {
    if (!provider || !model) return { pricing: null, source: 'unknown' };

    const override = getOverride(provider, model);
    if (override) return { pricing: override, source: 'user_override' };

    if (API_PRICING_PROVIDERS.has(provider)) {
        const fromOption = readPricingFromModelOption();
        if (fromOption) return { pricing: fromOption, source: 'provider_api' };
    }

    if (pricingDefaults && pricingDefaults[provider]) {
        const provData = pricingDefaults[provider];
        if (provData[model]) {
            const { input, output } = provData[model];
            return {
                pricing: { input, output },
                source: 'default_table',
            };
        }
        const lower = model.toLowerCase();
        for (const knownModel of Object.keys(provData)) {
            if (knownModel.toLowerCase() === lower) {
                const { input, output } = provData[knownModel];
                return { pricing: { input, output }, source: 'default_table' };
            }
        }
    }

    return { pricing: null, source: 'unknown' };
}

function formatUSD(amount) {
    if (amount === 0) return '$0.00';
    if (Math.abs(amount) < 0.01) return '<$0.01';
    if (Math.abs(amount) < 1) return `$${amount.toFixed(3)}`;
    return `$${amount.toFixed(2)}`;
}

function renderBadge(state) {
    const badge = DomHelpers.getElement('costEstimateBadge');
    if (!badge) return;

    badge.classList.remove(
        'cost-free',
        'cost-estimated',
        'cost-unknown',
        'cost-loading',
        'cost-error',
    );

    if (state.kind === 'hidden') {
        badge.style.display = 'none';
        badge.textContent = '';
        badge.title = '';
        return;
    }

    badge.style.display = 'flex';

    if (state.kind === 'free') {
        badge.classList.add('cost-free');
        badge.innerHTML = `
            <span class="cost-badge-icon material-symbols-outlined">verified</span>
            <span class="cost-badge-text">${state.message || 'Free (local)'}</span>
        `;
        badge.title = '';
        return;
    }

    if (state.kind === 'loading') {
        badge.classList.add('cost-loading');
        badge.innerHTML = `
            <span class="cost-badge-icon material-symbols-outlined cost-spinning">progress_activity</span>
            <span class="cost-badge-text">Estimating cost...</span>
        `;
        badge.title = '';
        return;
    }

    if (state.kind === 'unknown') {
        badge.classList.add('cost-unknown');
        const provider = state.provider || '';
        const model = state.model || '';
        badge.innerHTML = `
            <span class="cost-badge-icon material-symbols-outlined">help</span>
            <span class="cost-badge-text">Pricing not set for this model</span>
            <button type="button" class="cost-badge-edit" data-action="edit"
                title="Set custom prices for ${provider}/${model}">Set prices</button>
        `;
        badge.title = `${provider} / ${model}`;
        return;
    }

    if (state.kind === 'no_content') {
        badge.classList.add('cost-unknown');
        badge.innerHTML = `
            <span class="cost-badge-icon material-symbols-outlined">description</span>
            <span class="cost-badge-text">Add a file or text to estimate cost</span>
        `;
        badge.title = '';
        return;
    }

    if (state.kind === 'error') {
        badge.classList.add('cost-error');
        badge.innerHTML = `
            <span class="cost-badge-icon material-symbols-outlined">error</span>
            <span class="cost-badge-text">Estimation error</span>
        `;
        badge.title = state.message || '';
        return;
    }

    badge.classList.add('cost-estimated');
    const min = formatUSD(state.total_cost_min);
    const max = formatUSD(state.total_cost_max);
    const display = (min === max) ? `Estimated: ${min}` : `Estimated: ${min} – ${max}`;

    const passesNote = state.passes && state.passes > 1
        ? `, ${state.passes} passes`
        : '';
    const tokensNote = state.input_tokens
        ? `${state.input_tokens.toLocaleString()} input tokens, ${state.n_chunks} chunk(s)${passesNote}`
        : '';
    const sourceNote = sourceLabel(state.pricing_source, state.pricing_last_updated);

    badge.innerHTML = `
        <span class="cost-badge-icon material-symbols-outlined">payments</span>
        <span class="cost-badge-text">${display}</span>
        <button type="button" class="cost-badge-edit" data-action="edit" title="Edit prices">Edit</button>
    `;
    badge.title = [tokensNote, sourceNote].filter(Boolean).join(' • ');
}

function sourceLabel(source, lastUpdated) {
    switch (source) {
        case 'user_override': return 'Your custom prices';
        case 'provider_api':  return 'Live prices from provider API';
        case 'default_table': return `Default prices (updated ${lastUpdated || pricingLastUpdated || ''})`;
        default: return '';
    }
}

async function ensureDefaults() {
    if (pricingDefaults) return;
    try {
        const data = await ApiClient.getPricingDefaults();
        pricingDefaults = data?.pricing || {};
        pricingLastUpdated = data?.last_updated || null;
    } catch {
        pricingDefaults = {};
    }
}

export const CostEstimator = {
    initialize() {
        this.attachListeners();
        ensureDefaults().then(() => this.refresh());
    },

    attachListeners() {
        if (listenersAttached) return;
        listenersAttached = true;

        window.addEventListener('modelChanged', () => this.refresh());
        window.addEventListener('fileListChanged', () => this.refresh());
        window.addEventListener('translationOptionsChanged', () => this.refresh());

        const inputText = DomHelpers.getElement('inputText');
        if (inputText) {
            inputText.addEventListener('input', () => this.refreshDebounced());
        }

        ['refineTranslation', 'textCleanup', 'sourceLang', 'targetLang'].forEach((id) => {
            const el = DomHelpers.getElement(id);
            if (el) el.addEventListener('change', () => this.refresh());
        });

        const badge = DomHelpers.getElement('costEstimateBadge');
        if (badge) {
            badge.addEventListener('click', (event) => {
                const target = event.target.closest('[data-action="edit"]');
                if (target) {
                    event.preventDefault();
                    this.openEditModal();
                }
            });
        }
    },

    refreshDebounced() {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => this.refresh(), DEBOUNCE_MS);
    },

    async refresh() {
        const provider = getCurrentProvider();
        const model = getCurrentModel();

        if (!provider || !model) {
            renderBadge({ kind: 'hidden' });
            return;
        }

        if (LOCAL_PROVIDERS.has(provider)) {
            renderBadge({ kind: 'free', message: 'Free (local model)' });
            return;
        }

        const { pricing, source } = resolvePricing(provider, model);
        if (!pricing) {
            renderBadge({ kind: 'unknown', provider, model });
            return;
        }

        const text = getInputText();
        const filePath = getCurrentFilePath();
        if (!text.trim() && !filePath) {
            renderBadge({ kind: 'no_content' });
            return;
        }

        if (inFlightController) inFlightController.abort();
        inFlightController = new AbortController();

        renderBadge({ kind: 'loading' });

        const { src, tgt } = getLanguagePair();
        const payload = {
            provider,
            model,
            src_lang: src,
            tgt_lang: tgt,
            options: getOptions(),
            pricing,
        };
        if (text.trim()) payload.text = text;
        else if (filePath) payload.file_path = filePath;

        try {
            const data = await ApiClient.estimateCost(payload);
            if (data.free) {
                renderBadge({ kind: 'free', message: data.message });
                return;
            }
            if (data.unknown) {
                renderBadge({ kind: 'unknown', provider, model });
                return;
            }
            if (data.no_content) {
                renderBadge({ kind: 'no_content' });
                return;
            }
            renderBadge({
                kind: 'estimated',
                ...data,
                pricing_source: source,
            });
        } catch (error) {
            if (error?.name === 'AbortError') return;
            renderBadge({ kind: 'error', message: error?.message });
        }
    },

    openEditModal() {
        const provider = getCurrentProvider();
        const model = getCurrentModel();
        if (!provider || !model) return;

        const existing = document.getElementById('costPricingModal');
        if (existing) existing.remove();

        const current = resolvePricing(provider, model).pricing
            || readPricingFromModelOption()
            || { input: 0, output: 0 };

        const override = getOverride(provider, model);

        const modal = document.createElement('div');
        modal.id = 'costPricingModal';
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal-content cost-pricing-modal">
                <div class="modal-header">
                    <h3>Edit pricing</h3>
                    <button class="close-btn" data-action="close">&times;</button>
                </div>
                <div class="modal-body">
                    <p class="cost-pricing-subtitle">
                        ${DomHelpers.escapeHtml(provider)} / <strong>${DomHelpers.escapeHtml(model)}</strong>
                    </p>
                    <p class="cost-pricing-help">
                        Prices are in USD per 1 million tokens. These values are saved
                        locally in your browser and only used for the estimation badge.
                    </p>
                    <div class="cost-pricing-grid">
                        <div class="form-group">
                            <label for="costPriceInput">Input ($ / 1M tokens)</label>
                            <div class="neu-inset-light">
                                <input type="number" min="0" step="0.001"
                                    id="costPriceInput" class="form-control"
                                    value="${current.input}">
                            </div>
                        </div>
                        <div class="form-group">
                            <label for="costPriceOutput">Output ($ / 1M tokens)</label>
                            <div class="neu-inset-light">
                                <input type="number" min="0" step="0.001"
                                    id="costPriceOutput" class="form-control"
                                    value="${current.output}">
                            </div>
                        </div>
                    </div>
                    ${override ? '<p class="cost-pricing-note">A custom price is currently saved for this model.</p>' : ''}
                </div>
                <div class="modal-footer">
                    ${override ? '<button class="btn btn-secondary" data-action="reset">Reset to default</button>' : ''}
                    <button class="btn btn-secondary" data-action="close">Cancel</button>
                    <button class="btn btn-primary" data-action="save">Save</button>
                </div>
            </div>
        `;
        document.body.appendChild(modal);

        const close = () => modal.remove();

        modal.addEventListener('click', (event) => {
            if (event.target === modal) close();
            const action = event.target.closest('[data-action]')?.dataset.action;
            if (action === 'close') close();
            if (action === 'save') {
                const inputEl = modal.querySelector('#costPriceInput');
                const outputEl = modal.querySelector('#costPriceOutput');
                const inputVal = parseFloat(inputEl.value);
                const outputVal = parseFloat(outputEl.value);
                if (!Number.isFinite(inputVal) || inputVal < 0 ||
                    !Number.isFinite(outputVal) || outputVal < 0) {
                    inputEl.focus();
                    return;
                }
                setOverride(provider, model, { input: inputVal, output: outputVal });
                close();
                this.refresh();
            }
            if (action === 'reset') {
                clearOverride(provider, model);
                close();
                this.refresh();
            }
        });

        const onEsc = (e) => {
            if (e.key === 'Escape') {
                close();
                document.removeEventListener('keydown', onEsc);
            }
        };
        document.addEventListener('keydown', onEsc);
    },
};
