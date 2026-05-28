/**
 * Sample & Compare manager — orchestrates UI for the Sample tab.
 *
 * Owns:
 *  - upload state for the selected book file
 *  - the dynamic list of LLM columns (1..4)
 *  - kick-off + stop of a sample run via the backend
 *  - subscription to the `sample_update` WebSocket event for streaming cells
 *  - a cross-Run results cache so identical (item, llm, params) cells are
 *    never re-translated; adding / removing an LLM updates the displayed grid
 *    immediately and only the new column hits the backend on the next Run.
 *
 * Delegates rendering of the comparison grid to SampleTable. Inline diff
 * (translate+refine) lives in sample-diff.js.
 */

import { ApiClient } from '../core/api-client.js';
import { WebSocketManager } from '../core/websocket-manager.js';
import { DomHelpers } from '../ui/dom-helpers.js';
import { t, applyToDOM } from '../i18n/i18n.js';
import { SampleTable } from './sample-table.js';
import { SearchableSelectFactory } from '../ui/searchable-select.js';
import {
    PROVIDER_ORDER,
    PROVIDER_META,
    attachProviderSearchable,
    attachModelSearchable,
    populateModelSelectInto,
    setPlaceholderOption,
} from '../providers/provider-select-helpers.js';

// Track every SearchableSelect we attach inside the column cards so we can
// destroy them before re-rendering — `#sampleColumns` gets wiped on every
// render, which would otherwise leave orphan instances in the factory map.
const sampleSearchableIds = new Set();

const MAX_COLUMNS = 4;

const state = {
    file: null,          // File object
    uploadedPath: null,  // path on server after upload
    fileType: null,
    columns: [],         // [{provider, model, temperature, context_window}]
    mode: 'translate',
    currentSampleId: null,
    running: false,
    // Snapshot of the last Run — used by add/remove column to re-render the
    // result table without needing a fresh server roundtrip.
    lastItems: null,            // [{index, source_text, truncated}] or null before first Run
    lastRunContext: null,       // { mode, source_lang, target_lang, prompt_options, glossary_id }
    // Maps "row:col" -> cellKey for the *currently displayed* table, so that
    // streamed cell_done events know which cache key to update.
    currentRunKeys: new Map(),
    // Last (N, max_chars) actually fed to /api/sample/initialize; the "Update
    // samples" button is enabled only when the current input values differ
    // from these.
    appliedNSamples: null,
    appliedMaxChars: null,
};

// Cross-Run result cache. Persists across Runs for the lifetime of the page.
// Key: canonical JSON of (source_text, llm config, mode, langs, prompt_options, glossary_id)
// Value: { translate?: {output, metrics}, refine?: {output, metrics} }
const resultsCache = new Map();

function $(id) {
    return document.getElementById(id);
}

/**
 * Stable JSON for prompt_options — used as part of the cache key. Only the
 * fields that actually affect the prompt are included.
 */
function normalizePromptOptions(po) {
    return JSON.stringify({
        ci: (po && po.custom_instructions) || '',
        ptc: !!(po && po.preserve_technical_content),
        tc: !!(po && po.text_cleanup),
    });
}

/**
 * Build a canonical cache key for a (source extract, LLM config, run context)
 * tuple. Two cells share a key iff their LLM outputs are expected to match.
 */
function buildCellKey(item, col, runCtx) {
    return JSON.stringify({
        src: item.source_text,
        mode: runCtx.mode,
        sl: runCtx.source_lang || '',
        tl: runCtx.target_lang || '',
        provider: col.provider || '',
        model: col.model || '',
        temp: typeof col.temperature === 'number' ? col.temperature : 0.3,
        ctx: col.context_window || 0,
        po: normalizePromptOptions(runCtx.prompt_options),
        gl: runCtx.glossary_id || '',
    });
}

/**
 * Which phases must have a 'done' entry for a cached cell to count as a hit?
 */
function requiredPhases(mode) {
    if (mode === 'refine') return ['refine'];
    if (mode === 'translate_refine') return ['translate', 'refine'];
    return ['translate'];
}

function isFullCacheHit(entry, mode) {
    if (!entry) return false;
    return requiredPhases(mode).every((p) => entry[p] && entry[p].status === 'done');
}

/**
 * Show a status message inline within the Sample tab.
 *
 * `MessageLogger.showMessage` targets `#messages`, which lives inside the
 * Translate tab — so its toasts are invisible while the user is on Sample.
 * We render here instead so failures don't appear silent.
 */
function showSampleMessage(text, type = 'info') {
    const box = $('sampleWarnings');
    if (!box) return;
    const div = document.createElement('div');
    div.className = `sample-warning sample-warning-${type}`;
    const icon = type === 'error' ? '✖' : (type === 'success' ? '✓' : '⚠');
    div.textContent = `${icon} ${text}`;
    box.innerHTML = '';
    box.appendChild(div);
}

function setButtonsRunningState(running) {
    state.running = running;
    const runBtn = $('sampleRunBtn');
    const stopBtn = $('sampleStopBtn');
    const addBtn = $('sampleAddColumnBtn');
    if (runBtn) runBtn.disabled = running;
    if (stopBtn) stopBtn.classList.toggle('hidden', !running);
    if (addBtn) addBtn.disabled = running || state.columns.length >= MAX_COLUMNS;
    document.querySelectorAll('#sampleColumns .sample-column-remove').forEach((btn) => {
        btn.disabled = running || state.columns.length <= 1;
    });
    syncSampleEditButtons();
    syncUpdateButton();
}

/**
 * Disable the per-card "remove" and the "add a sample" buttons while a Run
 * is in flight — editing the sample set mid-run would race with arriving
 * WebSocket cell_done events.
 *
 * Called after every render of #sampleResults (whose markup is re-built
 * from scratch each time) so the disabled state always reflects state.running.
 */
function syncSampleEditButtons() {
    const running = state.running;
    const addBtn = $('sampleAddSampleBtn');
    if (addBtn) addBtn.disabled = running;
    document.querySelectorAll('#sampleResults .sample-card-remove').forEach((btn) => {
        btn.disabled = running;
    });
    // Visual feedback: pending cells in other columns can't be clicked while
    // one column is currently translating.
    document.querySelectorAll('#sampleResults .sample-cell-pending').forEach((el) => {
        el.classList.toggle('is-disabled', running);
    });
}

/**
 * Fetch + render models for a column. Uses the SAME backend path and the SAME
 * per-provider rendering (Gemini token tooltips, OpenRouter/Poe pricing
 * labels, Poe optgroups, …) as the Settings panel. With `__USE_ENV__` the
 * server resolves the API key from `.env`.
 *
 * Returns the first model's value when the column had no model yet, so the
 * caller can keep `col.model` in sync.
 */
async function loadAndPopulateModelsForColumn(col, modelSelectEl) {
    setPlaceholderOption(modelSelectEl, 'common:loading');
    try {
        const data = await ApiClient.getModels(col.provider, { apiKey: '__USE_ENV__' });
        const models = data.models || [];
        if (!models.length) {
            setPlaceholderOption(modelSelectEl, 'settings:search_models_no_models_available');
            col.model = '';
            return '';
        }
        populateModelSelectInto(modelSelectEl, models, col.model || data.default || '', col.provider);
        // populateModelSelectInto leaves the native <select> value pointing at
        // the matched option (or the first if nothing matched); mirror that
        // into the column state so the next Run uses the visible selection.
        const picked = modelSelectEl.value;
        col.model = picked;
        return picked;
    } catch (err) {
        console.error('[sample] model fetch failed', err);
        setPlaceholderOption(modelSelectEl, 'settings:search_models_error');
        col.model = '';
        return '';
    }
}

function buildColumnCard(idx, col) {
    const card = document.createElement('div');
    card.className = 'sample-column-card neu-card-light';
    card.style.padding = '12px';
    card.style.border = '1px solid var(--border-light, rgba(0,0,0,0.1))';
    card.style.borderRadius = '0.75rem';
    card.dataset.idx = String(idx);

    const providerId = `sampleColProvider-${idx}`;
    const modelId = `sampleColModel-${idx}`;

    const providerOptions = PROVIDER_ORDER.map((value) => {
        const meta = PROVIDER_META[value] || { name: value };
        const selected = col.provider === value ? 'selected' : '';
        return `<option value="${value}" ${selected}>${meta.name}</option>`;
    }).join('');

    card.innerHTML = `
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px;">
            <strong>${t('sample:llm_label', { index: idx + 1, defaultValue: `LLM ${idx + 1}` })}</strong>
            <button type="button" class="btn btn-secondary sample-column-remove" data-i18n-attr="title:sample:remove_llm" title="${t('sample:remove_llm')}" style="padding: 0.3rem 0.6rem; font-size: 0.8125rem;">
                <span class="material-symbols-outlined" style="font-size: 1rem;">delete</span>
            </button>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
            <div class="form-group" style="margin-bottom: 0;">
                <label data-i18n="settings:ai_provider">Provider</label>
                <select id="${providerId}" class="form-control sample-provider-select">${providerOptions}</select>
            </div>
            <div class="form-group" style="margin-bottom: 0;">
                <label data-i18n="settings:model">Model</label>
                <select id="${modelId}" class="form-control sample-model-select">
                    <option value="" data-i18n="common:loading">Loading...</option>
                </select>
            </div>
        </div>
    `;

    const removeBtn = card.querySelector('.sample-column-remove');
    removeBtn.addEventListener('click', () => removeColumn(idx));

    // Defer SearchableSelect setup until the card is in the DOM — the
    // factory inserts its wrapper next to the original <select>.
    setTimeout(async () => {
        const providerSelectEl = document.getElementById(providerId);
        const modelSelectEl = document.getElementById(modelId);
        if (!providerSelectEl || !modelSelectEl) return;

        attachProviderSearchable(providerSelectEl, {
            onChange: async (newProvider) => {
                col.provider = newProvider;
                col.model = '';
                await loadAndPopulateModelsForColumn(col, modelSelectEl);
                refreshResultsFromCache();
            },
        });
        sampleSearchableIds.add(providerId);

        attachModelSearchable(modelSelectEl, {
            onChange: (value) => {
                col.model = value;
                refreshResultsFromCache();
            },
        });
        sampleSearchableIds.add(modelId);

        await loadAndPopulateModelsForColumn(col, modelSelectEl);
        refreshResultsFromCache();
    }, 0);

    return card;
}

function renderColumns() {
    const container = $('sampleColumns');
    if (!container) return;
    // Tear down any SearchableSelect instances bound to the previous card
    // DOM before wiping innerHTML, otherwise the factory keeps references to
    // detached <select> elements (and to stale `col` closures pointing at
    // re-indexed columns).
    sampleSearchableIds.forEach((id) => SearchableSelectFactory.destroy(id));
    sampleSearchableIds.clear();
    container.innerHTML = '';
    state.columns.forEach((col, idx) => {
        container.appendChild(buildColumnCard(idx, col));
    });
    applyToDOM(container);
    const addBtn = $('sampleAddColumnBtn');
    if (addBtn) addBtn.disabled = state.columns.length >= MAX_COLUMNS;
    document.querySelectorAll('#sampleColumns .sample-column-remove').forEach((btn) => {
        btn.disabled = state.columns.length <= 1;
    });
}

/**
 * Re-render the sample table from current `state.lastItems` × `state.columns`,
 * pulling whatever is cached from `resultsCache` when a run context is known.
 *
 * Called in three situations:
 *   - right after /initialize, before any Run (no lastRunContext, all
 *     skeletons but the cards are shown)
 *   - after add/remove sample or add/remove column (some prefilled, some
 *     skeletons)
 *   - inside runSample once the server returns items (most prefilled if the
 *     user keeps the same config)
 *
 * No-op if no file has been initialized yet (`state.lastItems === null`).
 */
function refreshResultsFromCache() {
    if (state.lastItems === null) return;
    const results = $('sampleResults');
    if (!results) return;

    const prefilled = new Map();
    state.currentRunKeys = new Map();

    if (state.lastRunContext) {
        state.lastItems.forEach((item, rowIdx) => {
            state.columns.forEach((col, colIdx) => {
                const key = buildCellKey(item, col, state.lastRunContext);
                state.currentRunKeys.set(`${rowIdx}:${colIdx}`, key);
                const cached = resultsCache.get(key);
                if (cached) {
                    const merged = {};
                    if (cached.translate) merged.translate = { status: 'done', ...cached.translate };
                    if (cached.refine) merged.refine = { status: 'done', ...cached.refine };
                    if (Object.keys(merged).length > 0) {
                        prefilled.set(`${rowIdx}:${colIdx}`, merged);
                    }
                }
            });
        });
    }

    const mode = (state.lastRunContext && state.lastRunContext.mode) || state.mode;
    SampleTable.render(results, state.lastItems, state.columns, mode, { prefilled });
    applyToDOM(results);
    syncSampleEditButtons();
    const copyBtn = $('sampleCopyMdBtn');
    if (copyBtn) copyBtn.disabled = !SampleTable.hasResults();
}

function addColumn() {
    if (state.columns.length >= MAX_COLUMNS) return;
    state.columns.push({
        provider: 'ollama',
        model: '',
        temperature: 0.3,
        context_window: 4096,
    });
    renderColumns();
    refreshResultsFromCache();
}

function removeColumn(idx) {
    if (state.columns.length <= 1) return;
    state.columns.splice(idx, 1);
    renderColumns();
    refreshResultsFromCache();
}

function setMode(mode) {
    state.mode = mode;
    document.querySelectorAll('#sampleModeButtons .sample-mode-btn').forEach((btn) => {
        btn.classList.toggle('sample-mode-btn-active', btn.dataset.mode === mode);
    });
}

async function uploadFileIfNeeded() {
    if (state.uploadedPath) return state.uploadedPath;
    if (!state.file) {
        throw new Error(t('sample:error_no_file'));
    }
    const result = await ApiClient.uploadFile(state.file);
    state.uploadedPath = result.file_path;
    state.fileType = result.file_type;
    state.thumbnail = result.thumbnail || null;
    updateFileCard();
    return result.file_path;
}

/**
 * Refresh the selected-file card: cover (EPUB thumbnail or file-type icon),
 * filename, and a "size · type" details line. Called when a file is picked
 * and again once /api/upload returns a thumbnail.
 */
function updateFileCard() {
    if (!state.file) return;
    const nameEl = $('sampleFileName');
    const detailsEl = $('sampleFileDetails');
    const coverEl = $('sampleFileCover');
    if (nameEl) nameEl.textContent = state.file.name;

    if (detailsEl) {
        const sizeKb = state.file.size != null ? `${(state.file.size / 1024).toFixed(1)} KB` : '';
        const ext = (state.fileType || (state.file.name.split('.').pop() || '')).toUpperCase();
        detailsEl.textContent = [ext, sizeKb].filter(Boolean).join(' · ');
    }

    if (coverEl) {
        coverEl.innerHTML = '';
        if (state.thumbnail) {
            const img = document.createElement('img');
            img.src = `/api/thumbnails/${encodeURIComponent(state.thumbnail)}`;
            img.alt = '';
            img.onerror = () => {
                coverEl.innerHTML = `<span class="material-symbols-outlined">${iconForFileType(state.fileType)}</span>`;
            };
            coverEl.appendChild(img);
        } else {
            coverEl.innerHTML = `<span class="material-symbols-outlined">${iconForFileType(state.fileType)}</span>`;
        }
    }
}

function iconForFileType(ft) {
    const t = (ft || '').toLowerCase();
    if (t === 'epub') return 'menu_book';
    if (t === 'srt')  return 'closed_caption';
    if (t === 'docx') return 'description';
    if (t === 'txt')  return 'article';
    return 'description';
}

/**
 * Toggle between the dropzone (no file picked yet) and the rich file card
 * (file picked). Hiding the dropzone makes the selected book unmistakable
 * and prevents accidental drag-drop of a different file mid-session.
 */
function showFileCardMode(hasFile) {
    const dropzone = $('sampleFileUpload');
    const card = $('sampleFileInfo');
    if (hasFile) {
        if (dropzone) dropzone.classList.add('hidden');
        DomHelpers.show(card);
    } else {
        if (dropzone) dropzone.classList.remove('hidden');
        DomHelpers.hide(card);
    }
}

/**
 * Triggered after the user picks/drops a file. Uploads it (if needed) and
 * calls /api/sample/initialize so the sample cards appear immediately, before
 * any LLM call. The user can then curate them (X / Add) at no token cost.
 */
async function initializeSamples() {
    if (!state.file) return;
    const warningsBox = $('sampleWarnings');
    if (warningsBox) warningsBox.innerHTML = '';
    const results = $('sampleResults');
    if (results) {
        results.innerHTML = `<p class="sample-empty" data-i18n="sample:initializing">${t('sample:initializing')}</p>`;
    }
    state.lastItems = null;
    state.lastRunContext = null;
    state.currentSampleId = null;
    state.currentRunKeys = new Map();

    try {
        await uploadFileIfNeeded();
    } catch (err) {
        console.error('[sample] upload failed', err);
        showSampleMessage(err.message || String(err), 'error');
        return;
    }
    await _runInitialize({ preserveContext: false });
}

/**
 * Re-sample the already-uploaded document with the current (N, max_chars)
 * inputs. Triggered by the "Update samples" button. Unlike a fresh upload,
 * this preserves `state.lastRunContext` so cells whose source_text still
 * matches a cached translation reappear instantly — only the *changed*
 * positions show as pending.
 */
async function refreshSampleSet() {
    if (state.running) return;
    if (!state.file || !state.uploadedPath) return;
    const warningsBox = $('sampleWarnings');
    if (warningsBox) warningsBox.innerHTML = '';
    await _runInitialize({ preserveContext: true });
}

async function _runInitialize({ preserveContext }) {
    const warningsBox = $('sampleWarnings');
    const nSamples = parseInt($('sampleNSamples')?.value, 10) || 5;
    const maxChars = parseInt($('sampleMaxChars')?.value, 10) || 180;

    try {
        const r = await fetch('/api/sample/initialize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                file_path: state.uploadedPath,
                file_type: state.fileType,
                n_samples: nSamples,
                max_chars: maxChars,
            }),
        });
        const body = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(body.error || `HTTP ${r.status}`);
        state.lastItems = body.items || [];
        state.appliedNSamples = nSamples;
        state.appliedMaxChars = maxChars;
        if (!preserveContext) {
            state.lastRunContext = null;
        }
        if (warningsBox && body.warnings && body.warnings.length > 0) {
            warningsBox.innerHTML = body.warnings.map((w) =>
                `<div class="sample-warning">⚠ ${w}</div>`
            ).join('');
        }
        refreshResultsFromCache();
        syncUpdateButton();
    } catch (err) {
        console.error('[sample] initialize failed', err);
        showSampleMessage(t('sample:initialize_failed', { error: err.message || String(err) }), 'error');
        if (!preserveContext) {
            state.lastItems = null;
            refreshResultsFromCache();
        }
        syncUpdateButton();
    }
}

/**
 * Enable the "Update samples" button only when there's a loaded file, no Run
 * is in flight, AND at least one of the two number inputs differs from the
 * value that was last fed to /initialize.
 */
function syncUpdateButton() {
    const btn = $('sampleUpdateBtn');
    if (!btn) return;
    const nVal = parseInt($('sampleNSamples')?.value, 10);
    const mVal = parseInt($('sampleMaxChars')?.value, 10);
    const hasFile = !!state.uploadedPath;
    const dirty = (
        Number.isFinite(nVal) && Number.isFinite(mVal) &&
        (nVal !== state.appliedNSamples || mVal !== state.appliedMaxChars)
    );
    btn.disabled = !hasFile || state.running || !dirty;
}

function removeSample(rowIdx) {
    if (!Array.isArray(state.lastItems)) return;
    if (rowIdx < 0 || rowIdx >= state.lastItems.length) return;
    state.lastItems.splice(rowIdx, 1);
    refreshResultsFromCache();
}

async function addSample() {
    if (!state.uploadedPath || !state.fileType) {
        showSampleMessage(t('sample:error_no_file'), 'error');
        return;
    }
    const maxChars = parseInt($('sampleMaxChars')?.value, 10) || 180;
    const excludeIndices = Array.isArray(state.lastItems)
        ? state.lastItems.map((it) => it.index)
        : [];

    const addBtn = $('sampleAddSampleBtn');
    if (addBtn) addBtn.disabled = true;

    try {
        const r = await fetch('/api/sample/extract', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                file_path: state.uploadedPath,
                file_type: state.fileType,
                max_chars: maxChars,
                exclude_indices: excludeIndices,
            }),
        });
        const body = await r.json().catch(() => ({}));
        if (r.status === 409) {
            showSampleMessage(t('sample:no_more_indices'), 'info');
            return;
        }
        if (!r.ok) throw new Error(body.error || `HTTP ${r.status}`);

        if (!Array.isArray(state.lastItems)) state.lastItems = [];
        state.lastItems.push(body.item);
        refreshResultsFromCache();
    } catch (err) {
        console.error('[sample] extract failed', err);
        showSampleMessage(t('sample:add_sample_failed', { error: err.message || String(err) }), 'error');
    } finally {
        const btn = $('sampleAddSampleBtn');
        if (btn) btn.disabled = false;
    }
}

function buildRunPayload({ defer = false } = {}) {
    const sourceLang = $('sampleSourceLang')?.value || '';
    const targetLang = $('sampleTargetLang')?.value || '';
    const nSamples = parseInt($('sampleNSamples')?.value, 10) || 5;
    const maxChars = parseInt($('sampleMaxChars')?.value, 10) || 800;

    const columns = state.columns.map((col) => ({
        provider: col.provider,
        model: col.model,
        temperature: col.temperature,
        context_window: col.context_window,
        api_key: '__USE_ENV__',
    }));

    const promptOptions = {
        custom_instructions: $('customInstructionsTextarea')?.value || '',
        preserve_technical_content: $('preserveTechnicalContent')?.checked || false,
        text_cleanup: $('textCleanup')?.checked || false,
    };

    const payload = {
        file_path: state.uploadedPath,
        file_type: state.fileType,
        source_language: sourceLang,
        target_language: targetLang,
        mode: state.mode,
        n_samples: nSamples,
        max_chars: maxChars,
        columns,
        prompt_options: promptOptions,
        glossary_id: $('glossarySelect')?.value || null,
        defer_dispatch: defer,
    };

    // The sample set is owned by the client once /initialize has run; pass it
    // along so the server doesn't re-shuffle behind the user's back.
    if (Array.isArray(state.lastItems)) {
        payload.items = state.lastItems.map((it) => ({
            index: it.index,
            source_text: it.source_text,
            truncated: !!it.truncated,
        }));
    }

    return payload;
}

/**
 * Kick off an LLM run.
 *
 * `opts.onlyColumn` (number) restricts the run to a single LLM column —
 * useful for the "click a grey cell to translate this column only" shortcut,
 * which avoids unload/reload churn on local providers like Ollama. When
 * omitted, every column is eligible (the global Run button).
 */
async function runSample(opts = {}) {
    if (state.running) return;
    const onlyColumn = (typeof opts.onlyColumn === 'number' && opts.onlyColumn >= 0)
        ? opts.onlyColumn
        : null;

    const warningsBox = $('sampleWarnings');
    if (warningsBox) warningsBox.innerHTML = '';

    if (state.columns.length === 0) {
        showSampleMessage(t('sample:error_no_llms'), 'error');
        return;
    }
    if (onlyColumn !== null) {
        if (onlyColumn >= state.columns.length) return;
        const targetCol = state.columns[onlyColumn];
        if (!targetCol || !targetCol.model) {
            showSampleMessage(t('sample:error_llm_missing_model', { index: onlyColumn + 1 }), 'error');
            return;
        }
    } else {
        const missingModelIdx = state.columns.findIndex((c) => !c.model);
        if (missingModelIdx !== -1) {
            showSampleMessage(t('sample:error_llm_missing_model', { index: missingModelIdx + 1 }), 'error');
            return;
        }
    }
    if (!state.file) {
        showSampleMessage(t('sample:error_no_file'), 'error');
        return;
    }
    if (!Array.isArray(state.lastItems) || state.lastItems.length === 0) {
        showSampleMessage(t('sample:error_no_samples'), 'error');
        return;
    }

    try {
        setButtonsRunningState(true);
        await uploadFileIfNeeded();
    } catch (err) {
        console.error('[sample] upload failed', err);
        showSampleMessage(err.message || String(err), 'error');
        setButtonsRunningState(false);
        return;
    }

    // Phase 1 — server samples items and creates the state entry, but does NOT
    // yet spend any LLM tokens. We need the actual extracts before we can know
    // which cells are cached.
    const payload = buildRunPayload({ defer: true });
    console.log('[sample] POST /api/sample/run (defer)', payload);

    let resp;
    try {
        const r = await fetch('/api/sample/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        resp = await r.json().catch(() => ({}));
        if (!r.ok) {
            throw new Error(resp.error || `HTTP ${r.status}`);
        }
    } catch (err) {
        console.error('[sample] /api/sample/run failed', err);
        showSampleMessage(t('sample:error_run', { error: err.message || String(err) }), 'error');
        setButtonsRunningState(false);
        return;
    }

    state.currentSampleId = resp.sample_id;
    state.lastItems = resp.items;
    state.lastRunContext = {
        mode: resp.mode,
        source_lang: payload.source_language,
        target_lang: payload.target_language,
        prompt_options: payload.prompt_options,
        glossary_id: payload.glossary_id,
    };

    // Phase 2 — compute cache hits, render the table (cached cells show their
    // content immediately, cells the server will work on now show the animated
    // shimmer, the rest fall back to the static "Click Run" hint), and tell
    // the server which (row, col) pairs to skip.
    const prefilled = new Map();
    const runningCells = new Set();
    const skipCells = [];
    state.currentRunKeys = new Map();

    resp.items.forEach((item, rowIdx) => {
        resp.columns.forEach((col, colIdx) => {
            const key = buildCellKey(item, col, state.lastRunContext);
            state.currentRunKeys.set(`${rowIdx}:${colIdx}`, key);
            const cached = resultsCache.get(key);
            const cellKey = `${rowIdx}:${colIdx}`;

            const mergedFromCache = () => {
                const merged = {};
                if (cached.translate) merged.translate = { status: 'done', ...cached.translate };
                if (cached.refine) merged.refine = { status: 'done', ...cached.refine };
                return merged;
            };

            // Off-target columns in a partial Run stay frozen — show what's
            // cached, leave the rest as the static pending hint, never run.
            if (onlyColumn !== null && colIdx !== onlyColumn) {
                if (isFullCacheHit(cached, resp.mode)) {
                    prefilled.set(cellKey, mergedFromCache());
                }
                skipCells.push([rowIdx, colIdx]);
                return;
            }

            if (isFullCacheHit(cached, resp.mode)) {
                prefilled.set(cellKey, mergedFromCache());
                skipCells.push([rowIdx, colIdx]);
            } else {
                runningCells.add(cellKey);
            }
        });
    });

    SampleTable.render($('sampleResults'), resp.items, resp.columns, resp.mode, { prefilled, runningCells });
    applyToDOM($('sampleResults'));
    syncSampleEditButtons();
    $('sampleCopyMdBtn').disabled = false;

    if (warningsBox && resp.warnings && resp.warnings.length > 0) {
        warningsBox.innerHTML = resp.warnings.map((w) =>
            `<div class="sample-warning">⚠ ${w}</div>`
        ).join('');
    }

    // Phase 3 — kick off the LLM work for non-cached cells. Server will emit
    // sample_done once finished (including when every cell is skipped).
    try {
        const r = await fetch(`/api/sample/${encodeURIComponent(resp.sample_id)}/dispatch`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ skip_cells: skipCells }),
        });
        if (!r.ok) {
            const errBody = await r.json().catch(() => ({}));
            throw new Error(errBody.error || `HTTP ${r.status}`);
        }
    } catch (err) {
        console.error('[sample] /dispatch failed', err);
        showSampleMessage(t('sample:error_run', { error: err.message || String(err) }), 'error');
        setButtonsRunningState(false);
    }
}

async function stopSample() {
    if (!state.currentSampleId) return;
    try {
        await fetch(`/api/sample/${encodeURIComponent(state.currentSampleId)}/stop`, { method: 'POST' });
    } catch (err) {
        console.warn('stop sample failed', err);
    }
}

function copyAsMarkdown() {
    const md = SampleTable.toMarkdown();
    if (!md) return;
    navigator.clipboard.writeText(md).then(
        () => showSampleMessage(t('sample:copied_to_clipboard'), 'success'),
        (err) => showSampleMessage(t('sample:copy_failed', { error: err.message || String(err) }), 'error'),
    );
}

function reset() {
    state.file = null;
    state.uploadedPath = null;
    state.fileType = null;
    state.thumbnail = null;
    state.currentSampleId = null;
    state.lastItems = null;
    state.lastRunContext = null;
    state.currentRunKeys = new Map();
    showFileCardMode(false);
    const results = $('sampleResults');
    if (results) {
        results.innerHTML = `<p data-i18n="sample:no_results_yet">${t('sample:no_results_yet')}</p>`;
    }
    $('sampleCopyMdBtn').disabled = true;
}

function handleSampleUpdate(payload) {
    if (!payload || payload.sample_id !== state.currentSampleId) return;

    if (payload.type === 'cell_done' || payload.type === 'cell_error') {
        SampleTable.updateCell(payload);
        if (payload.type === 'cell_done') {
            const key = state.currentRunKeys.get(`${payload.row}:${payload.col}`);
            if (key) {
                const entry = resultsCache.get(key) || {};
                // `status: 'done'` is required by isFullCacheHit() so the next
                // Run treats this cell as a cache hit and skips the LLM call.
                entry[payload.phase] = {
                    status: 'done',
                    output: payload.output,
                    metrics: payload.metrics,
                };
                resultsCache.set(key, entry);
            }
        }
        return;
    }

    if (payload.type === 'sample_done' || payload.type === 'sample_stopped') {
        setButtonsRunningState(false);
        showSampleMessage(
            payload.type === 'sample_done'
                ? t('sample:run_done')
                : t('sample:run_stopped'),
            payload.type === 'sample_done' ? 'success' : 'info',
        );
    }
}

function onFileSelected(file) {
    state.file = file;
    state.uploadedPath = null;
    state.fileType = null;
    state.thumbnail = null;
    updateFileCard();
    showFileCardMode(true);
    initializeSamples();
}

/**
 * Clear the selected file: reset all sample state, switch the UI back to the
 * dropzone, and wipe the result table. Triggered by the X button on the file
 * card. No-op while a Run is in flight.
 */
function clearSelectedFile() {
    if (state.running) return;
    state.file = null;
    state.uploadedPath = null;
    state.fileType = null;
    state.thumbnail = null;
    state.lastItems = null;
    state.lastRunContext = null;
    state.currentSampleId = null;
    state.currentRunKeys = new Map();
    state.appliedNSamples = null;
    state.appliedMaxChars = null;
    const fileInput = $('sampleFileInput');
    if (fileInput) fileInput.value = '';
    showFileCardMode(false);
    const warningsBox = $('sampleWarnings');
    if (warningsBox) warningsBox.innerHTML = '';
    const results = $('sampleResults');
    if (results) {
        results.innerHTML = `<p data-i18n="sample:no_results_yet">${t('sample:no_results_yet')}</p>`;
        applyToDOM(results);
    }
    const copyBtn = $('sampleCopyMdBtn');
    if (copyBtn) copyBtn.disabled = true;
    syncUpdateButton();
}

function wireFileInput() {
    const input = $('sampleFileInput');
    const uploadZone = $('sampleFileUpload');

    if (input) {
        input.addEventListener('change', (e) => {
            const f = e.target.files && e.target.files[0];
            if (!f) return;
            onFileSelected(f);
        });
    }

    if (uploadZone) {
        ['dragover', 'dragenter'].forEach((evt) => {
            uploadZone.addEventListener(evt, (e) => {
                e.preventDefault();
                uploadZone.classList.add('drag-over');
            });
        });
        ['dragleave', 'drop'].forEach((evt) => {
            uploadZone.addEventListener(evt, (e) => {
                e.preventDefault();
                uploadZone.classList.remove('drag-over');
            });
        });
        uploadZone.addEventListener('drop', (e) => {
            const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
            if (!f) return;
            onFileSelected(f);
        });
    }
}

function wireResultsDelegation() {
    const results = $('sampleResults');
    if (!results) return;
    // SampleTable rebuilds the DOM on every render, so we delegate clicks
    // from the stable #sampleResults container instead of binding per-card.
    results.addEventListener('click', (e) => {
        const removeBtn = e.target.closest('.sample-card-remove');
        if (removeBtn) {
            const row = parseInt(removeBtn.dataset.row, 10);
            if (!Number.isNaN(row)) removeSample(row);
            return;
        }
        if (e.target.closest('#sampleAddSampleBtn')) {
            addSample();
            return;
        }
        // Click a grey "pending" cell to translate that LLM column only —
        // keeps the model loaded for every sample in one pass.
        const pending = e.target.closest('.sample-cell-pending');
        if (pending && !state.running) {
            const block = pending.closest('.sample-llm-block');
            if (block) {
                const colIdx = parseInt(block.dataset.col, 10);
                if (!Number.isNaN(colIdx)) {
                    runSample({ onlyColumn: colIdx });
                }
            }
        }
    });
}

function wireButtons() {
    $('sampleAddColumnBtn')?.addEventListener('click', addColumn);
    $('sampleRunBtn')?.addEventListener('click', runSample);
    $('sampleStopBtn')?.addEventListener('click', stopSample);
    $('sampleCopyMdBtn')?.addEventListener('click', copyAsMarkdown);
    $('sampleFileRemoveBtn')?.addEventListener('click', clearSelectedFile);
    $('sampleUpdateBtn')?.addEventListener('click', refreshSampleSet);

    // Live-enable the Update button as the user tweaks N or max_chars.
    ['sampleNSamples', 'sampleMaxChars'].forEach((id) => {
        $(id)?.addEventListener('input', syncUpdateButton);
    });

    document.querySelectorAll('#sampleModeButtons .sample-mode-btn').forEach((btn) => {
        btn.addEventListener('click', () => setMode(btn.dataset.mode));
    });

    wireResultsDelegation();
}

function rerenderOnLocale() {
    window.addEventListener('localeChanged', () => {
        renderColumns();
        if (SampleTable.hasResults()) {
            applyToDOM($('sampleResults'));
        }
    });
}

export const SampleManager = {
    init() {
        if (state.columns.length === 0) addColumn();
        renderColumns();
        wireFileInput();
        wireButtons();
        rerenderOnLocale();
        WebSocketManager.on('sample_update', handleSampleUpdate);
        setMode('translate');
    },
    addColumn,
    removeColumn,
    run: runSample,
    stop: stopSample,
    copyAsMarkdown,
    reset,
};
