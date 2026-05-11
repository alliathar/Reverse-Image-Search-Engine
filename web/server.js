const express = require('express');
const multer = require('multer');
const { spawn } = require('child_process');
const path = require('path');
const cors = require('cors');
const fs = require('fs');
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

// API endpoint for search
app.post('/api/search', upload.single('queryImage'), async (req, res) => {
    if (!req.file) return res.status(400).json({ error: 'Missing query image.' });

    const queryImagePath = req.file.path;
    const category = req.body.category || 'all';

    try {
        const line = await sendCommand(`SEARCH ${queryImagePath} ${category}`);
        fs.unlink(queryImagePath, () => {});
        res.json(JSON.parse(line));
    } catch (e) {
        fs.unlink(queryImagePath, () => {});
        res.status(500).json({ error: 'Failed to process image search.' });
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
