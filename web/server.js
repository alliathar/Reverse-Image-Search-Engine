const express = require('express');
const multer = require('multer');
const { spawn } = require('child_process');
const path = require('path');
const cors = require('cors');
const fs = require('fs');
const os = require('os');
const readline = require('readline');

const app = express();
const port = 0;

app.use(cors());
app.use(express.static('public')); // Serve the frontend
app.use(express.json());
// Multer config for query image upload
const uploadDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadDir)) {
    fs.mkdirSync(uploadDir);
}

const upload = multer({ dest: 'uploads/' });
// Path to compiled C++ Executable
const exeName = process.platform === 'win32' ? 'ReverseImageSearch.exe' : 'ReverseImageSearch';
const exePath = path.join(__dirname, '..', 'build', exeName);    

let pendingResolve = null;
let queue = Promise.resolve();

const engine = spawn(exePath);

const rl = readline.createInterface({ input: engine.stdout });
rl.on('line', (line) => {
    if (pendingResolve) {
        const resolve = pendingResolve;
        pendingResolve = null;
        resolve(line);
    }
});
engine.stderr.on('data', (d) => console.error('[engine]', d.toString()));
engine.on('exit', (code) => console.error('[engine] exited with code', code));

function sendCommand(cmd) {
    return new Promise((resolve, reject) => {
        queue = queue.then(() => new Promise((done) => {
            pendingResolve = (line) => { done(); resolve(line); };
            engine.stdin.write(cmd + '\n');
        }));
        queue.catch(reject);
    });
}

app.post('/api/load', async (req, res) => {
    const { datasetPath } = req.body;
    if (!datasetPath) return res.status(400).json({ error: 'Missing datasetPath.' });
    try {
        const line = await sendCommand(`LOAD ${datasetPath}`);
        res.json(JSON.parse(line));
    } catch (e) {
        res.status(500).json({ error: 'Failed to load dataset.' });
    }
});

// API endpoint for search. Accepts up to 8 files for multi-image mode.
app.post('/api/search', upload.array('queryImage', 8), async (req, res) => {
    if (!req.files || req.files.length === 0) {
        return res.status(400).json({ error: 'Missing query image.' });
    }

    const queryImagePaths = req.files.map(f => f.path);
    const mode = req.body.mode || 'normal';     // normal | negative | multi

    // Categories may arrive as a comma-separated string from the multi-state dropdown.
    // Empty / missing → "_" sentinel (meaning "all" for include, "none" for exclude).
    const sanitize = v => (v && v.trim()) ? v.trim() : '_';
    const include = sanitize(req.body.include);
    const exclude = sanitize(req.body.exclude);

    // Protocol: SEARCH <mode> <include_csv> <exclude_csv> <path1> [<path2>...]
    const cmd = `SEARCH ${mode} ${include} ${exclude} ${queryImagePaths.join(' ')}`;
    const cleanup = () => queryImagePaths.forEach(p => fs.unlink(p, () => {}));

    try {
        const line = await sendCommand(cmd);
        cleanup();
        res.json(JSON.parse(line));
    } catch (e) {
        cleanup();
        res.status(500).json({ error: 'Failed to process image search.' });
    }
});


// API endpoint: list subfolders of a given directory for the folder picker.
// Defaults to the user's home directory.
// Find the most useful "Downloads" folder, preferring Windows Downloads under WSL.
function defaultBrowsePath() {
    const candidates = [];

    // 1. WSL: Windows Downloads via the user's WSL login name first.
    if (fs.existsSync('/mnt/c/Users')) {
        const me = process.env.USER || process.env.USERNAME;
        if (me) candidates.push(`/mnt/c/Users/${me}/Downloads`);
        // 2. Scan /mnt/c/Users for any non-system user folder that has Downloads.
        try {
            const skip = new Set(['Public', 'Default', 'Default User', 'All Users']);
            for (const name of fs.readdirSync('/mnt/c/Users')) {
                if (skip.has(name) || name.startsWith('.')) continue;
                candidates.push(`/mnt/c/Users/${name}/Downloads`);
            }
        } catch {}
    }
    // 3. The native ~/Downloads (Linux, macOS, native Windows).
    candidates.push(path.join(os.homedir(), 'Downloads'));
    // 4. Last resort.
    candidates.push(os.homedir());

    for (const p of candidates) {
        try {
            if (!fs.statSync(p).isDirectory()) continue;
            // Make sure we can actually list it — otherwise the browse endpoint
            // will EACCES on the very first open.
            fs.accessSync(p, fs.constants.R_OK | fs.constants.X_OK);
            return p;
        } catch {}
    }
    return os.homedir();
}

app.get('/api/browse', (req, res) => {
    let target = req.query.path && req.query.path.trim();
    if (!target) target = defaultBrowsePath();
    try {
        const resolved = path.resolve(target);
        const stat = fs.statSync(resolved);
        if (!stat.isDirectory()) {
            return res.status(400).json({ error: 'Not a directory' });
        }
        const entries = fs.readdirSync(resolved, { withFileTypes: true });
        const folders = entries
            .filter(e => e.isDirectory() && !e.name.startsWith('.'))
            .map(e => e.name)
            .sort((a, b) => a.localeCompare(b));
        const parent = path.dirname(resolved);
        res.json({
            path: resolved,
            parent: parent === resolved ? null : parent,
            folders,
        });
    } catch (e) {
        res.status(400).json({ error: e.message });
    }
});

// API endpoint to serve arbitrary local target images to the frontend securely
app.get('/api/image', (req, res) => {
    const targetPath = req.query.path;
    if (!targetPath) {
        return res.status(400).send('No path provided');
    }
    res.sendFile(path.resolve(targetPath), (err) => {
        if (err) {
            res.status(404).send('Image not found');
        }
    });
});

const server = app.listen(port, () => {
    const actualPort = server.address().port;
    console.log(`Server running at http://localhost:${actualPort}`);
});
