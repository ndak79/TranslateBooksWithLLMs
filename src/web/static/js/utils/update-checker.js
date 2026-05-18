/**
 * UpdateChecker - shows an update banner when a newer GitHub release exists
 * and drives the self-update flow (git pull + pip install + server restart).
 *
 * Endpoints used:
 *   GET  /api/version/check
 *   POST /api/version/update
 *   GET  /api/version/update/status (polled while update runs)
 */

import { MessageLogger } from '../ui/message-logger.js';

const DISMISS_STORAGE_KEY = 'tbl_update_dismissed_version';
const POLL_INTERVAL_MS = 1500;
const RESTART_WAIT_MAX_MS = 90_000;

export const UpdateChecker = {
    _pollTimer: null,
    _restartWaiter: null,
    _initialHealthSessionId: null,

    async initialize() {
        this._wireBannerHandlers();
        this._wireOverlayHandlers();
        try {
            await this.checkOnce();
        } catch (e) {
            console.warn('Initial version check failed:', e);
        }
    },

    _wireBannerHandlers() {
        const installBtn = document.getElementById('updateBannerInstallBtn');
        const closeBtn = document.getElementById('updateBannerCloseBtn');
        if (installBtn) {
            installBtn.addEventListener('click', () => this.startUpdate());
        }
        if (closeBtn) {
            closeBtn.addEventListener('click', () => this.dismissBanner());
        }
    },

    _wireOverlayHandlers() {
        const closeBtn = document.getElementById('updateOverlayCloseBtn');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => this.hideOverlay());
        }
    },

    async checkOnce() {
        const resp = await fetch('/api/version/check');
        if (!resp.ok) return null;
        const data = await resp.json();
        this._applyCheckResult(data);
        return data;
    },

    _applyCheckResult(data) {
        const banner = document.getElementById('updateBanner');
        if (!banner) return;

        if (!data || !data.update_available || !data.latest) {
            banner.classList.add('hidden');
            return;
        }

        const dismissed = localStorage.getItem(DISMISS_STORAGE_KEY);
        if (dismissed && dismissed === String(data.latest)) {
            banner.classList.add('hidden');
            return;
        }

        const versionEl = document.getElementById('updateBannerVersion');
        const linkEl = document.getElementById('updateBannerLink');
        const installBtn = document.getElementById('updateBannerInstallBtn');

        if (versionEl) {
            versionEl.textContent = ` v${data.latest} is available (you have v${data.current}).`;
        }
        if (linkEl && data.release_url) {
            linkEl.href = data.release_url;
        }
        if (installBtn) {
            // Disable the in-place update button when the install is not a git
            // checkout (e.g. zip download, PyInstaller bundle). The release-
            // notes link remains usable so the user can still update manually.
            if (data.git_repo === false) {
                installBtn.disabled = true;
                installBtn.title = 'Auto-update needs a git checkout. Use the release link to update manually.';
            } else {
                installBtn.disabled = false;
                installBtn.title = '';
            }
        }

        banner.classList.remove('hidden');
    },

    dismissBanner() {
        const banner = document.getElementById('updateBanner');
        const versionEl = document.getElementById('updateBannerVersion');
        if (banner) banner.classList.add('hidden');
        if (versionEl) {
            const match = /v([0-9][\w.\-]*)/.exec(versionEl.textContent || '');
            if (match) {
                localStorage.setItem(DISMISS_STORAGE_KEY, match[1]);
            }
        }
    },

    async startUpdate() {
        const installBtn = document.getElementById('updateBannerInstallBtn');
        if (installBtn) installBtn.disabled = true;

        this.showOverlay();
        this._setOverlayStep('Sending update request...');

        try {
            const resp = await fetch('/api/version/update', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ install_deps: true, restart: true }),
            });
            const data = await resp.json().catch(() => ({}));

            if (!resp.ok) {
                const reason = data.message || data.error || `HTTP ${resp.status}`;
                this._setOverlayError(reason);
                if (installBtn) installBtn.disabled = false;
                MessageLogger.addLog(`Update refused: ${reason}`);
                return;
            }

            MessageLogger.addLog('Update started.');
            this._captureCurrentSessionId();
            this._startPolling();
        } catch (e) {
            this._setOverlayError(`Failed to contact server: ${e.message}`);
            if (installBtn) installBtn.disabled = false;
        }
    },

    async _captureCurrentSessionId() {
        try {
            const r = await fetch('/api/health');
            const j = await r.json();
            this._initialHealthSessionId = String(j.session_id || j.startup_time || '');
        } catch {
            this._initialHealthSessionId = null;
        }
    },

    _startPolling() {
        this._stopPolling();
        const tick = async () => {
            try {
                const r = await fetch('/api/version/update/status');
                if (!r.ok) return;
                const status = await r.json();
                this._applyStatus(status);

                if (status.state === 'failed') {
                    this._stopPolling();
                    return;
                }
                if (status.state === 'completed') {
                    if (status.requires_restart) {
                        this._setOverlayStep('Restarting server...');
                        this._stopPolling();
                        this._waitForRestart();
                    } else {
                        this._stopPolling();
                        this._setOverlayStep('Update complete. Restart the app manually to apply.');
                        this._showOverlayCloseButton();
                    }
                }
            } catch (e) {
                // Network blip during restart is expected; let _waitForRestart handle it.
            }
        };
        tick();
        this._pollTimer = setInterval(tick, POLL_INTERVAL_MS);
    },

    _stopPolling() {
        if (this._pollTimer) {
            clearInterval(this._pollTimer);
            this._pollTimer = null;
        }
    },

    _waitForRestart() {
        const deadline = Date.now() + RESTART_WAIT_MAX_MS;
        const probe = async () => {
            try {
                const r = await fetch('/api/health', { cache: 'no-store' });
                if (r.ok) {
                    const j = await r.json();
                    const newSession = String(j.session_id || j.startup_time || '');
                    if (newSession && newSession !== this._initialHealthSessionId) {
                        this._setOverlayStep(`Server restarted on v${j.version}. Reloading...`);
                        clearInterval(this._restartWaiter);
                        this._restartWaiter = null;
                        setTimeout(() => window.location.reload(), 700);
                        return;
                    }
                }
            } catch {
                // server is down during restart, keep probing
            }
            if (Date.now() > deadline) {
                clearInterval(this._restartWaiter);
                this._restartWaiter = null;
                this._setOverlayError(
                    'Server did not come back automatically. If you launched the app with start.bat / start.sh ' +
                    'it should restart on its own; otherwise relaunch it manually.'
                );
                this._showOverlayCloseButton();
            }
        };
        this._restartWaiter = setInterval(probe, 2000);
        probe();
    },

    _applyStatus(status) {
        if (!status) return;
        if (status.step) {
            this._setOverlayStep(status.step);
        }
        const log = document.getElementById('updateOverlayLog');
        if (log && Array.isArray(status.output)) {
            log.textContent = status.output.join('\n');
            log.scrollTop = log.scrollHeight;
        }
        if (status.state === 'failed') {
            this._setOverlayError(status.error || 'Update failed.');
            this._showOverlayCloseButton();
        }
    },

    showOverlay() {
        const ov = document.getElementById('updateOverlay');
        const err = document.getElementById('updateOverlayError');
        const log = document.getElementById('updateOverlayLog');
        const closeBtn = document.getElementById('updateOverlayCloseBtn');
        if (ov) ov.classList.remove('hidden');
        if (err) { err.style.display = 'none'; err.textContent = ''; }
        if (log) log.textContent = '';
        if (closeBtn) closeBtn.style.display = 'none';
    },

    hideOverlay() {
        const ov = document.getElementById('updateOverlay');
        if (ov) ov.classList.add('hidden');
    },

    _setOverlayStep(text) {
        const el = document.getElementById('updateOverlayStep');
        if (el) el.textContent = text;
    },

    _setOverlayError(message) {
        const el = document.getElementById('updateOverlayError');
        if (el) {
            el.textContent = message;
            el.style.display = 'block';
        }
    },

    _showOverlayCloseButton() {
        const btn = document.getElementById('updateOverlayCloseBtn');
        if (btn) btn.style.display = 'inline-flex';
    },
};
