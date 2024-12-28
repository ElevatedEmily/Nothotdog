const functions = require("firebase-functions");
const express = require("express");
const cors = require("cors");
const multer = require("multer");
const { exec } = require("child_process");

const app = express();
app.use(cors({ origin: true }));
const upload = multer();

app.post("/predict", upload.single("file"), async (req, res) => {
  try {
    const fileBuffer = req.file.buffer;

    // Call your PyTorch model or script here
    exec("python ../app.py", (error, stdout, stderr) => {
      if (error) {
        return res.status(500).send({ error: stderr });
      }
      return res.status(200).send({ prediction: stdout.trim() });
    });
  } catch (err) {
    res.status(500).send({ error: err.message });
  }
});

exports.api = functions.https.onRequest(app);
