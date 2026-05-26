import React, { useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import * as tf from '@tensorflow/tfjs';
import {
  Activity,
  BrainCircuit,
  Camera,
  Download,
  FlaskConical,
  Image,
  Upload,
  RotateCcw,
  Save,
  Trash2,
  Volume2
} from 'lucide-react';
import './styles.css';

const FEATURE_VERSION = 'wrist-relative-mirror-v1';
const DEFAULT_SIGNS = [
  'A',
  'B',
  'C',
  'D',
  'E',
  'F',
  'G',
  'H',
  'I',
  'J',
  'K',
  'L',
  'M',
  'N',
  'O',
  'P',
  'Q',
  'R',
  'S',
  'T',
  'U',
  'V',
  'W',
  'X',
  'Y'
];
const REMOVED_DEFAULT_SIGNS = ['Ä', 'Ö', 'Ü', 'CH', 'SCH', 'Z'];
const TARGET_PER_SIGN = 20;
const DATASET_KEY = 'dgs-landmark-dataset-v1';
const LABELS_KEY = 'dgs-landmark-labels-v1';
const CAMERA_KEY = 'dgs-camera-device-id-v1';
const REFERENCE_IMAGE_URL = 'https://www.sgb-fss.ch/content/uploads/2024/03/fingeralphabet-a6-d-vorderseite-logo.jpg';
const LETTER_IMAGE_BASE = '/Dataset';
const MODEL_NAME = 'dgs-mlp-v1';
const MODEL_META_KEY = 'dgs-model-meta-v1';
const DEFAULT_MODEL_BUNDLE_URL = '/dgs-mlp-v1-bundle.json';
const APP_NAME = 'DGS Fingeralphabet Trainer';
const LANGUAGE_KEY = 'dgs-practice-language-v1';
const HOLD_TO_CONFIRM_MS = 1000;
const USER_COPY = {
  de: {
    navPractice: 'Üben',
    navData: 'Daten',
    navTraining: 'Training',
    navStory: 'Story',
    noHand: 'Keine Hand',
    loading: 'Lädt',
    handDetected: 'Hand erkannt',
    modelActive: 'Modell aktiv',
    noModel: 'Kein Modell',
    practice: 'Übung',
    detected: 'Erkanntes Zeichen',
    confidence: 'Sicherheit',
    free: 'Frei',
    practiceButton: 'Üben',
    speak: 'Sprechen',
    newWord: 'Neues Wort',
    ownModel: 'Eigenes Modell',
    wordDone: 'Geschafft!',
    wordDoneSubtitle: (word) => `Du hast „${word}" gebärdet`,
    celebrate: (word) => `Super! ${word}`,
    historyEyebrow: 'Projektverlauf',
    historyTitle: 'Vom Datensatz zur Übung',
    historyText: 'Diese Seite ist die Hauptansicht zum Üben. Das Standardmodell ist bereits enthalten. Wer möchte, kann eigene Handdaten aufnehmen, ein persönliches Modell trainieren und es hier über',
    stepDataTitle: '1. Daten aufnehmen',
    stepDataText: 'Zeichen wählen, Hand zeigen und JSON-Datensätze exportieren.',
    stepTrainTitle: '2. Modell trainieren',
    stepTrainText: 'Mehrere JSON-Dateien kombinieren und ein neues Bundle erzeugen.',
    stepPracticeTitle: '3. Wörter üben',
    stepPracticeText: 'Das Zeichen halten, bis der Fortschrittsbalken voll ist.'
  },
  en: {
    navPractice: 'Practice',
    navData: 'Data',
    navTraining: 'Training',
    navStory: 'Story',
    noHand: 'No hand',
    loading: 'Loading',
    handDetected: 'Hand detected',
    modelActive: 'Model active',
    noModel: 'No model',
    practice: 'Practice',
    detected: 'Detected sign',
    confidence: 'confidence',
    free: 'Free',
    practiceButton: 'Practice',
    speak: 'Speak',
    newWord: 'New word',
    ownModel: 'Own model',
    wordDone: 'Nice!',
    wordDoneSubtitle: (word) => `You signed "${word}"`,
    celebrate: (word) => `Nice! ${word}`,
    historyEyebrow: 'Project History',
    historyTitle: 'From dataset to practice',
    historyText: 'This is the main practice view. A default model is already included. Anyone can collect their own hand data, train a personal model, and upload it here through',
    stepDataTitle: '1. Collect data',
    stepDataText: 'Choose a sign, show your hand, and export JSON datasets.',
    stepTrainTitle: '2. Train model',
    stepTrainText: 'Combine multiple JSON files and create a new bundle.',
    stepPracticeTitle: '3. Practice words',
    stepPracticeText: 'Hold the sign until the progress bar is full.'
  }
};
const APP_COPY = {
  de: {
    subtitle: 'Üben, Daten sammeln und ein eigenes Modell trainieren',
    navPractice: 'Üben',
    navData: 'Daten',
    navTraining: 'Training'
  },
  en: {
    subtitle: 'Practice, collect data, and train your own model',
    navPractice: 'Practice',
    navData: 'Data',
    navTraining: 'Training'
  }
};
const DATA_COPY = {
  de: {
    ready: 'Bereit zum Aufnehmen',
    added: (label) => `1 Aufnahme für ${label} hinzugefügt`,
    capturing: (label) => `${label} wird aufgenommen`,
    captured: (count, label) => `${count}/${TARGET_PER_SIGN} Aufnahmen für ${label}`,
    done: (label, count, nextLabel) => nextLabel && nextLabel !== label
      ? `Fertig: ${label} +${count}. Weiter: ${nextLabel}`
      : `Fertig: ${label} +${count}`,
    noHand: 'Keine Hand erkannt, bitte erneut versuchen',
    empty: 'Lokaler Datensatz geleert',
    deleted: (label) => `Daten für ${label} gelöscht`,
    dataset: 'Datensatz',
    title: 'DGS-Aufnahme',
    capture20: '20 aufnehmen',
    capturingButton: 'Aufnahme läuft',
    captureOne: (label) => `${label} aufnehmen`,
    clearOne: (label) => `${label} leeren`,
    clearAll: 'Alles leeren',
    localFile: 'Lokale Datei',
    exportTitle: 'Datensatz exportieren',
    readySigns: 'Fertige Zeichen',
    samples: 'Aufnahmen',
    downloadJson: 'JSON herunterladen',
    noPhotos: 'Es werden keine Fotos gespeichert.',
    storageText: 'Der Browser speichert nur normalisierte Landmark-Koordinaten. Danach kann die JSON-Datei fürs Training genutzt werden.',
    nextTitle: 'Nächster Schritt',
    nextText: 'Wenn genug Daten aufgenommen sind, kannst du den Datensatz exportieren und direkt das Modell trainieren.',
    trainModel: 'Modell trainieren',
    help: 'Hilfe',
    howTo: (label) => `So geht ${label || '...'}`
  },
  en: {
    ready: 'Ready to record',
    added: (label) => `1 recording added for ${label}`,
    capturing: (label) => `Recording ${label}`,
    captured: (count, label) => `${count}/${TARGET_PER_SIGN} recordings for ${label}`,
    done: (label, count, nextLabel) => nextLabel && nextLabel !== label
      ? `Done: ${label} +${count}. Next: ${nextLabel}`
      : `Done: ${label} +${count}`,
    noHand: 'No hand detected, please try again',
    empty: 'Local dataset cleared',
    deleted: (label) => `Data for ${label} deleted`,
    dataset: 'Dataset',
    title: 'DGS recording',
    capture20: 'Record 20',
    capturingButton: 'Recording',
    captureOne: (label) => `Record ${label}`,
    clearOne: (label) => `Clear ${label}`,
    clearAll: 'Clear all',
    localFile: 'Local file',
    exportTitle: 'Export dataset',
    readySigns: 'Ready signs',
    samples: 'Recordings',
    downloadJson: 'Download JSON',
    noPhotos: 'No photos are saved.',
    storageText: 'The browser only stores normalized landmark coordinates. Use the JSON file for model training afterwards.',
    nextTitle: 'Next step',
    nextText: 'After recording enough data, export the dataset and continue to train the model.',
    trainModel: 'Train model',
    help: 'Help',
    howTo: (label) => `How to sign ${label || '...'}`
  }
};
const MODEL_COPY = {
  de: {
    ready: 'Bereit fürs Training',
    combined: (count, samples, labels) => `${count} JSON-Dateien kombiniert: ${samples} Aufnahmen, ${labels} Zeichen`,
    loadError: (error) => `Fehler beim Laden: ${error}`,
    preparing: 'Daten werden vorbereitet',
    trainingStart: (epochs) => `Training 0/${epochs}`,
    epoch: (epoch, epochs, loss, valAcc) => `Epoche ${epoch}/${epochs} - loss ${loss} - val_acc ${valAcc}%`,
    done: (acc) => `Fertig - val_acc ${acc}% - im Browser gespeichert`,
    error: (error) => `Fehler: ${error}`,
    dataset: 'Datensatz',
    title: 'Verfügbare Daten',
    signs: 'Zeichen',
    samples: 'Aufnahmen',
    minPerSign: 'Min / Zeichen',
    upload: 'JSONs hochladen',
    missingTitle: 'Daten fehlen',
    missingText: (labels) => `Keine Aufnahmen für: ${labels}. Nimm zuerst Daten auf oder lade exportierte JSON-Dateien hoch.`,
    train: 'Modell trainieren',
    training: 'Training läuft',
    download: 'Bundle herunterladen',
    progress: 'Fortschritt',
    loss: 'Loss-Kurve',
    perSign: 'Pro Zeichen',
    accuracy: 'Genauigkeit pro Zeichen'
  },
  en: {
    ready: 'Ready to train',
    combined: (count, samples, labels) => `${count} JSON files combined: ${samples} recordings, ${labels} signs`,
    loadError: (error) => `Loading error: ${error}`,
    preparing: 'Preparing data',
    trainingStart: (epochs) => `Training 0/${epochs}`,
    epoch: (epoch, epochs, loss, valAcc) => `Epoch ${epoch}/${epochs} - loss ${loss} - val_acc ${valAcc}%`,
    done: (acc) => `Done - val_acc ${acc}% - saved in the browser`,
    error: (error) => `Error: ${error}`,
    dataset: 'Dataset',
    title: 'Available data',
    signs: 'Signs',
    samples: 'Recordings',
    minPerSign: 'Min / sign',
    upload: 'Upload JSONs',
    missingTitle: 'Missing data',
    missingText: (labels) => `No recordings for: ${labels}. Record data first or upload exported JSON files.`,
    train: 'Train model',
    training: 'Training',
    download: 'Download bundle',
    progress: 'Progress',
    loss: 'Loss curve',
    perSign: 'Per sign',
    accuracy: 'Accuracy per sign'
  }
};
const PRACTICE_WORDS = [
  'HAUS',
  'HUND',
  'AUTO',
  'BALL',
  'MAMA',
  'OMA',
  'MOND',
  'MILCH',
  'MUND',
  'BOOT',
  'BUCH',
  'DACH',
  'DORF',
  'EIS',
  'FISCH',
  'HAND',
  'HUT',
  'KIND',
  'LILA',
  'ROSA',
  'BLAU',
  'ROT',
  'REIS',
  'TEE',
  'SAFT',
  'OFEN',
  'DOSE',
  'SOFA',
  'TISCH',
  'TOR',
  'TAG',
  'NACHT',
  'WALD',
  'WIND',
  'MAUS',
  'MALER',
  'LAMPE',
  'AMPEL',
  'BLUME',
  'BAUM',
  'RAUM',
  'NAME',
  'SALAMI',
  'GELB',
  'GRUEN',
  'KLEIN',
  'ALT',
  'NEU',
  'LEISE',
  'LAUT',
  'FEIN',
  'DUNKEL',
  'HELL',
  'REGEN',
  'RAD',
  'BUS',
  'BROT',
  'KASE',
  'APFEL',
  'OPA',
  'PAPA',
  'PARK',
  'POST',
  'PILOT',
  'PULS',
  'PLAN',
  'PAAR',
  'PFEIL'
];

function normalizeLandmarks(landmarks, handedness) {
  if (!landmarks?.length) return null;

  const mirror = handedness === 'Left' ? -1 : 1;
  const wrist = landmarks[0];
  const xs = landmarks.map((point) => point.x);
  const ys = landmarks.map((point) => point.y);
  const zs = landmarks.map((point) => point.z);
  const spanX = Math.max(...xs) - Math.min(...xs);
  const spanY = Math.max(...ys) - Math.min(...ys);
  const spanZ = Math.max(...zs) - Math.min(...zs);
  const scale = Math.max(spanX, spanY, spanZ, 0.001);

  return landmarks.flatMap((point) => [
    (mirror * (point.x - wrist.x)) / scale,
    (point.y - wrist.y) / scale,
    (point.z - wrist.z) / scale
  ]);
}

function downloadJson(filename, payload) {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function arrayBufferToBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  let binary = '';
  const chunkSize = 0x8000;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
  }
  return window.btoa(binary);
}

function base64ToArrayBuffer(base64) {
  const binary = window.atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes.buffer;
}

async function exportModelBundle(model, metadata) {
  let artifacts = null;
  await model.save(tf.io.withSaveHandler(async (nextArtifacts) => {
    artifacts = nextArtifacts;
    return {
      modelArtifactsInfo: {
        dateSaved: new Date(),
        modelTopologyType: 'JSON',
        weightDataBytes: nextArtifacts.weightData?.byteLength ?? 0
      }
    };
  }));

  downloadJson(`${MODEL_NAME}-bundle.json`, {
    bundleVersion: 1,
    ...metadata,
    modelTopology: artifacts.modelTopology,
    weightSpecs: artifacts.weightSpecs,
    weightDataBase64: arrayBufferToBase64(artifacts.weightData)
  });
}

async function loadModelBundle(bundle) {
  if (!bundle?.modelTopology || !Array.isArray(bundle?.weightSpecs) || !bundle?.weightDataBase64) {
    throw new Error('Die Datei ist kein gültiges Modell-Bundle.');
  }

  return tf.loadLayersModel(tf.io.fromMemory({
    modelTopology: bundle.modelTopology,
    weightSpecs: bundle.weightSpecs,
    weightData: base64ToArrayBuffer(bundle.weightDataBase64)
  }));
}

function loadStoredJson(key, fallback) {
  try {
    const value = window.localStorage.getItem(key);
    return value ? JSON.parse(value) : fallback;
  } catch {
    return fallback;
  }
}

function usePracticeLanguage() {
  const [language, setLanguage] = useState(() => window.localStorage.getItem(LANGUAGE_KEY) || 'de');

  function selectLanguage(nextLanguage) {
    setLanguage(nextLanguage);
    window.localStorage.setItem(LANGUAGE_KEY, nextLanguage);
  }

  return { language, selectLanguage };
}

function LanguageControl({ language, onChange }) {
  return (
    <label className="languageSelect">
      <span>Language</span>
      <select value={language} onChange={(event) => onChange(event.target.value)}>
        <option value="de">Deutsch</option>
        <option value="en">English</option>
      </select>
    </label>
  );
}

function mergeDefaultLabels(storedLabels) {
  const existing = Array.isArray(storedLabels) && storedLabels.length
    ? storedLabels.filter((label) => !REMOVED_DEFAULT_SIGNS.includes(label))
    : [];
  const merged = Array.from(new Set([...existing, ...DEFAULT_SIGNS]));
  return merged.sort((a, b) => a.localeCompare(b, 'de'));
}

function useHandTracker() {
  const videoRef = useRef(null);
  const landmarkerRef = useRef(null);
  const rafRef = useRef(null);
  const featuresRef = useRef(null);
  const [status, setStatus] = useState('Handerkennung wird geladen');
  const [ready, setReady] = useState(false);
  const [features, setFeatures] = useState(null);
  const [landmarks, setLandmarks] = useState([]);
  const [handedness, setHandedness] = useState(null);
  const [devices, setDevices] = useState([]);
  const [deviceId, setDeviceId] = useState(() => window.localStorage.getItem(CAMERA_KEY) || '');

  useEffect(() => {
    let cancelled = false;
    let stream = null;

    async function setup() {
      try {
        setReady(false);
        setStatus('Kamera wird geöffnet');
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: 960,
            height: 720,
            ...(deviceId ? { deviceId: { exact: deviceId } } : { facingMode: 'user' })
          },
          audio: false
        });

        if (cancelled) return;
        const video = videoRef.current;
        video.srcObject = stream;
        await video.play();

        const vision = await FilesetResolver.forVisionTasks('/mediapipe/wasm');
        const landmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: '/mediapipe/models/hand_landmarker.task'
          },
          numHands: 1,
          runningMode: 'VIDEO'
        });

        if (cancelled) {
          landmarker.close();
          return;
        }

        landmarkerRef.current = landmarker;
        setReady(true);
        setStatus('Handerkennung bereit');
        const mediaDevices = await navigator.mediaDevices.enumerateDevices();
        setDevices(mediaDevices.filter((device) => device.kind === 'videoinput'));

        function tick() {
          const result = landmarker.detectForVideo(video, performance.now());
          const hand = result.landmarks?.[0] ?? null;
          const handednessLabel =
            result.handednesses?.[0]?.[0]?.categoryName ??
            result.handedness?.[0]?.[0]?.categoryName ??
            null;
          const nextFeatures = hand ? normalizeLandmarks(hand, handednessLabel) : null;
          setLandmarks(hand ?? []);
          setFeatures(nextFeatures);
          setHandedness(hand ? handednessLabel : null);
          featuresRef.current = nextFeatures;
          rafRef.current = window.requestAnimationFrame(tick);
        }

        tick();
      } catch (error) {
        setStatus(error?.message || 'Kamera konnte nicht gestartet werden');
      }
    }

    setup();

    return () => {
      cancelled = true;
      if (rafRef.current) window.cancelAnimationFrame(rafRef.current);
      if (landmarkerRef.current) landmarkerRef.current.close();
      if (stream) stream.getTracks().forEach((track) => track.stop());
    };
  }, [deviceId]);

  function selectDevice(nextDeviceId) {
    setDeviceId(nextDeviceId);
    window.localStorage.setItem(CAMERA_KEY, nextDeviceId);
  }

  return { videoRef, ready, status, features, featuresRef, landmarks, handedness, devices, deviceId, selectDevice };
}

function CameraPanel({ tracker, badge }) {
  return (
    <section className="cameraPanel">
      <div className="videoShell">
        <video ref={tracker.videoRef} muted playsInline />
        <LandmarkOverlay landmarks={tracker.landmarks} />
        <div className={tracker.features ? 'handBadge live' : 'handBadge'}>
          <Activity size={15} />
          {tracker.features
            ? `Hand erkannt${tracker.handedness ? ` (${tracker.handedness === 'Left' ? 'Links' : 'Rechts'})` : ''}`
            : 'Keine Hand'}
        </div>
      </div>
      <div className="cameraMeta">
        <span>{tracker.status}</span>
        {badge && <strong>{badge}</strong>}
      </div>
      <div className="cameraControls">
        <label>
          Kamera
          <select value={tracker.deviceId} onChange={(event) => tracker.selectDevice(event.target.value)}>
            <option value="">Default</option>
            {tracker.devices.map((device, index) => (
              <option key={device.deviceId} value={device.deviceId}>
                {device.label || `Kamera ${index + 1}`}
              </option>
            ))}
          </select>
        </label>
      </div>
    </section>
  );
}

function LandmarkOverlay({ landmarks }) {
  if (!landmarks?.length) return null;

  return (
    <svg className="landmarks" viewBox="0 0 1 1" preserveAspectRatio="none" aria-hidden="true">
      {landmarks.map((point, index) => (
        <circle key={index} cx={point.x} cy={point.y} r="0.008" />
      ))}
    </svg>
  );
}

function HandFrame({ landmarks }) {
  if (!landmarks?.length) return null;

  const xs = landmarks.map((p) => p.x);
  const ys = landmarks.map((p) => p.y);
  const padX = 0.045;
  const padY = 0.06;
  const minX = Math.max(0, Math.min(...xs) - padX);
  const maxX = Math.min(1, Math.max(...xs) + padX);
  const minY = Math.max(0, Math.min(...ys) - padY);
  const maxY = Math.min(1, Math.max(...ys) + padY);
  const w = maxX - minX;
  const h = maxY - minY;
  const corner = Math.min(w, h) * 0.18;

  return (
    <svg className="handFrame" viewBox="0 0 1 1" preserveAspectRatio="none" aria-hidden="true">
      <rect
        className="handFrame-box"
        x={minX}
        y={minY}
        width={w}
        height={h}
        rx="0.022"
        ry="0.022"
      />
      <g className="handFrame-corners">
        <path d={`M ${minX} ${minY + corner} L ${minX} ${minY} L ${minX + corner} ${minY}`} />
        <path d={`M ${maxX - corner} ${minY} L ${maxX} ${minY} L ${maxX} ${minY + corner}`} />
        <path d={`M ${minX} ${maxY - corner} L ${minX} ${maxY} L ${minX + corner} ${maxY}`} />
        <path d={`M ${maxX - corner} ${maxY} L ${maxX} ${maxY} L ${maxX} ${maxY - corner}`} />
      </g>
      <g className="handFrame-dots">
        {landmarks.map((point, index) => (
          <circle key={index} cx={point.x} cy={point.y} r="0.0035" />
        ))}
      </g>
    </svg>
  );
}

function countByLabel(samples, labels) {
  return labels.reduce((acc, label) => {
    acc[label] = samples.filter((sample) => sample.label === label).length;
    return acc;
  }, {});
}

function getNextLabel(labels, currentLabel, counts) {
  const currentIndex = labels.indexOf(currentLabel);
  const orderedLabels = [
    ...labels.slice(currentIndex + 1),
    ...labels.slice(0, Math.max(currentIndex, 0))
  ];
  return orderedLabels.find((label) => (counts[label] ?? 0) < TARGET_PER_SIGN) ?? labels[currentIndex + 1] ?? labels[0] ?? '';
}

function TrainerApp() {
  const { language, selectLanguage } = usePracticeLanguage();
  const copy = DATA_COPY[language] ?? DATA_COPY.de;
  const tracker = useHandTracker();
  const [labels, setLabels] = useState(() => mergeDefaultLabels(loadStoredJson(LABELS_KEY, DEFAULT_SIGNS)));
  const [samples, setSamples] = useState(() => loadStoredJson(DATASET_KEY, []));
  const [selectedLabel, setSelectedLabel] = useState(() => mergeDefaultLabels(loadStoredJson(LABELS_KEY, DEFAULT_SIGNS))[0] ?? DEFAULT_SIGNS[0]);
  const [captureStatus, setCaptureStatus] = useState(() => copy.ready);
  const [burstCapturing, setBurstCapturing] = useState(false);

  const counts = useMemo(() => countByLabel(samples, labels), [samples, labels]);
  const completedLetters = labels.filter((label) => counts[label] >= TARGET_PER_SIGN).length;

  useEffect(() => {
    window.localStorage.setItem(DATASET_KEY, JSON.stringify(samples));
  }, [samples]);

  useEffect(() => {
    window.localStorage.setItem(LABELS_KEY, JSON.stringify(labels));
  }, [labels]);

  function makeSample(label, currentFeatures) {
    return {
      id: crypto.randomUUID(),
      label,
      features: currentFeatures,
      createdAt: new Date().toISOString()
    };
  }

  function addSample() {
    if (!tracker.features) return;
    setSamples((value) => [...value, makeSample(selectedLabel, tracker.features)]);
    setCaptureStatus(copy.added(selectedLabel));
  }

  async function captureTwenty() {
    if (burstCapturing) return;
    setBurstCapturing(true);
    setCaptureStatus(copy.capturing(selectedLabel));

    let added = 0;
    let misses = 0;
    let finalCounts = counts;

    while (added < TARGET_PER_SIGN && misses < 30) {
      const currentFeatures = tracker.featuresRef.current;

      if (currentFeatures) {
        let complete = false;
        setSamples((value) => {
          const currentCount = value.filter((sample) => sample.label === selectedLabel).length;
          if (currentCount >= TARGET_PER_SIGN) {
            complete = true;
            return value;
          }
          added += 1;
          const nextSamples = [...value, makeSample(selectedLabel, currentFeatures)];
          finalCounts = countByLabel(nextSamples, labels);
          return nextSamples;
        });
        if (complete) break;
      } else {
        misses += 1;
      }

      setCaptureStatus(copy.captured(added, selectedLabel));
      await new Promise((resolve) => window.setTimeout(resolve, 260));
    }

    if (added) {
      const nextLabel = getNextLabel(labels, selectedLabel, finalCounts);
      setSelectedLabel(nextLabel);
      setCaptureStatus(copy.done(selectedLabel, added, nextLabel));
    } else {
      setCaptureStatus(copy.noHand);
    }
    setBurstCapturing(false);
  }

  function exportDataset() {
    downloadJson('dgs-dataset.json', {
      featureVersion: FEATURE_VERSION,
      labels,
      targetPerSign: TARGET_PER_SIGN,
      exportedAt: new Date().toISOString(),
      counts,
      samples
    });
  }

  function resetDataset() {
    setSamples([]);
    setCaptureStatus(copy.empty);
  }

  function resetSelectedLetter() {
    setSamples((value) => value.filter((sample) => sample.label !== selectedLabel));
    setCaptureStatus(copy.deleted(selectedLabel));
  }

  return (
    <main className="app">
      <AppHeader active="train" language={language} onLanguageChange={selectLanguage} />
      <div className="workspace">
        <CameraPanel tracker={tracker} badge={`${samples.length} Aufnahmen`} />

        <section className="panel">
          <div className="panelHead">
            <div>
              <span>{copy.dataset}</span>
              <h2>{copy.title}</h2>
            </div>
            <button className={burstCapturing ? 'primary active' : 'primary'} disabled={!tracker.ready || burstCapturing} onClick={captureTwenty}>
              <Camera size={17} />
              {burstCapturing ? copy.capturingButton : copy.capture20}
            </button>
          </div>

          <div className="labelGrid">
            {labels.map((label) => (
              <button
                key={label}
                className={selectedLabel === label ? 'signButton selected' : 'signButton'}
                onClick={() => setSelectedLabel(label)}
              >
                <strong>{label}</strong>
                <span>{counts[label] ?? 0}/{TARGET_PER_SIGN}</span>
                <small style={{ width: `${Math.min(100, ((counts[label] ?? 0) / TARGET_PER_SIGN) * 100)}%` }} />
              </button>
            ))}
          </div>

          <div className="actions">
            <button className="primary" disabled={!tracker.features} onClick={addSample}>
              <Save size={17} />
              {copy.captureOne(selectedLabel)}
            </button>
            <button disabled={!counts[selectedLabel]} onClick={resetSelectedLetter}>
              <Trash2 size={17} />
              {copy.clearOne(selectedLabel)}
            </button>
            <button disabled={!samples.length} onClick={resetDataset}>
              <RotateCcw size={17} />
              {copy.clearAll}
            </button>
          </div>

          <div className="statusBox">
            <span>{captureStatus}</span>
          </div>
        </section>

        <div className="rightStack">
          <section className="panel">
            <div className="panelHead">
              <div>
                <span>{copy.localFile}</span>
                <h2>{copy.exportTitle}</h2>
              </div>
              <Download size={26} />
            </div>

            <div className="metricGrid">
              <Metric label={copy.readySigns} value={`${completedLetters}/${labels.length}`} />
              <Metric label={copy.samples} value={samples.length} />
              <Metric label="Features" value="63" />
            </div>

            <div className="actions">
              <button className="primary" disabled={!samples.length} onClick={exportDataset}>
                <Download size={17} />
                {copy.downloadJson}
              </button>
            </div>

            <div className="storageNote">
              <strong>{copy.noPhotos}</strong>
              <span>{copy.storageText}</span>
            </div>

            <div className="storageNote nextStepNote">
              <strong>{copy.nextTitle}</strong>
              <span>{copy.nextText}</span>
              <a className="textLinkButton" href="#/model">
                <BrainCircuit size={16} />
                {copy.trainModel}
              </a>
            </div>
          </section>

          <ReferencePanel selectedLabel={selectedLabel} copy={copy} />
        </div>
      </div>
    </main>
  );
}

function ReferencePanel({ selectedLabel, copy = DATA_COPY.de }) {
  const [errored, setErrored] = useState(false);
  const letterImage = selectedLabel ? `${LETTER_IMAGE_BASE}/${selectedLabel}.png` : '';

  useEffect(() => {
    setErrored(false);
  }, [selectedLabel]);

  return (
    <section className="panel referencePanel">
      <div className="panelHead">
        <div>
          <span>{copy.help}</span>
          <h2>{copy.howTo(selectedLabel)}</h2>
        </div>
        <Image size={26} />
      </div>
      {letterImage && !errored ? (
        <img
          className="letterImage"
          src={letterImage}
          alt={`Hilfe für ${selectedLabel}`}
          onError={() => setErrored(true)}
        />
      ) : (
        <a className="alphabetFallback" href={REFERENCE_IMAGE_URL} target="_blank" rel="noreferrer">
          <img src={REFERENCE_IMAGE_URL} alt="Alfabeto manual DGS/SGB-FSS" />
        </a>
      )}
    </section>
  );
}

function AppHeader({ active = 'train', language = 'de', onLanguageChange = () => {} }) {
  const copy = APP_COPY[language] ?? APP_COPY.de;
  return (
    <header className="topbar">
      <a className="brand" href="#/app">
        <FlaskConical size={30} />
        <div>
          <h1>{APP_NAME}</h1>
          <span>{copy.subtitle}</span>
        </div>
      </a>
      <nav>
        <a href="#/app" className={active === 'app' ? 'active' : ''}>{copy.navPractice}</a>
        <a href="#/train" className={active === 'train' ? 'active' : ''}>{copy.navData}</a>
        <a href="#/model" className={active === 'model' ? 'active' : ''}>{copy.navTraining}</a>
      </nav>
      <LanguageControl language={language} onChange={onLanguageChange} />
    </header>
  );
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function useHashRoute() {
  const [hash, setHash] = useState(() => window.location.hash || '#/app');
  useEffect(() => {
    function onHashChange() {
      setHash(window.location.hash || '#/app');
    }
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);
  return hash;
}

function splitDataset(samples, labels) {
  const trainX = [];
  const trainY = [];
  const valX = [];
  const valY = [];
  const byLabel = Object.fromEntries(labels.map((label) => [label, []]));
  samples.forEach((sample) => {
    if (byLabel[sample.label]) byLabel[sample.label].push(sample);
  });
  labels.forEach((label, labelIdx) => {
    const items = [...byLabel[label]];
    for (let i = items.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [items[i], items[j]] = [items[j], items[i]];
    }
    const valCount = Math.max(1, Math.floor(items.length * 0.2));
    items.slice(0, valCount).forEach((sample) => {
      valX.push(sample.features);
      valY.push(labelIdx);
    });
    items.slice(valCount).forEach((sample) => {
      trainX.push(sample.features);
      trainY.push(labelIdx);
    });
  });
  return { trainX, trainY, valX, valY };
}

function normalizeImportedDataset(payload) {
  const rawSamples = Array.isArray(payload?.samples) ? payload.samples : Array.isArray(payload) ? payload : [];
  if (!rawSamples.length) {
    throw new Error('Die JSON-Datei hat keine Aufnahmen in "samples".');
  }

  const labelsFromFile = Array.isArray(payload?.labels) ? payload.labels.map(String) : [];
  const labelsFromSamples = rawSamples.map((sample) => String(sample?.label ?? '')).filter(Boolean);
  const labels = Array.from(new Set([...labelsFromFile, ...labelsFromSamples]))
    .filter((label) => !REMOVED_DEFAULT_SIGNS.includes(label))
    .sort((a, b) => a.localeCompare(b, 'de'));

  const samples = rawSamples.map((sample, index) => {
    const label = String(sample?.label ?? '');
    const features = sample?.features;
    if (!label) throw new Error(`Aufnahme ${index + 1}: Label fehlt.`);
    if (!Array.isArray(features) || features.length !== 63) {
      throw new Error(`Aufnahme ${index + 1}: features muss 63 Zahlen haben.`);
    }
    if (!features.every((value) => Number.isFinite(Number(value)))) {
      throw new Error(`Aufnahme ${index + 1}: features enthält ungültige Werte.`);
    }
    return {
      id: sample.id || crypto.randomUUID(),
      label,
      features: features.map(Number),
      createdAt: sample.createdAt || new Date().toISOString()
    };
  });

  return { labels, samples };
}

function mergeImportedDatasets(datasets) {
  const labels = Array.from(new Set(datasets.flatMap((dataset) => dataset.labels)))
    .filter((label) => !REMOVED_DEFAULT_SIGNS.includes(label))
    .sort((a, b) => a.localeCompare(b, 'de'));
  const samples = datasets.flatMap((dataset) => dataset.samples);
  return { labels, samples };
}

function LossChart({ history }) {
  if (!history.length) {
    return <div className="chartPlaceholder">Sin datos aun</div>;
  }
  const W = 320;
  const H = 140;
  const padding = 10;
  const allValues = history.flatMap((entry) => [entry.loss, entry.val_loss ?? entry.loss]);
  const maxLoss = Math.max(...allValues);
  const minLoss = Math.min(...allValues);
  const range = Math.max(0.0001, maxLoss - minLoss);
  function toPath(values) {
    return values
      .map((value, index) => {
        const x = padding + (index / Math.max(1, history.length - 1)) * (W - 2 * padding);
        const y = H - padding - ((value - minLoss) / range) * (H - 2 * padding);
        return `${index === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`;
      })
      .join(' ');
  }
  const trainPath = toPath(history.map((entry) => entry.loss));
  const valPath = history[0]?.val_loss != null ? toPath(history.map((entry) => entry.val_loss)) : null;
  return (
    <div className="lossChartWrap">
      <svg className="lossChart" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
        <path d={trainPath} stroke="#42d392" fill="none" strokeWidth="1.5" />
        {valPath && <path d={valPath} stroke="#ffb199" fill="none" strokeWidth="1.5" />}
      </svg>
      <div className="lossLegend">
        <span><i style={{ background: '#42d392' }} /> train</span>
        <span><i style={{ background: '#ffb199' }} /> val</span>
      </div>
    </div>
  );
}

function TrainModelApp() {
  const { language, selectLanguage } = usePracticeLanguage();
  const copy = MODEL_COPY[language] ?? MODEL_COPY.de;
  const fileInputRef = useRef(null);
  const [samples, setSamples] = useState(() => loadStoredJson(DATASET_KEY, []));
  const [labels, setLabels] = useState(() => mergeDefaultLabels(loadStoredJson(LABELS_KEY, DEFAULT_SIGNS)));
  const [training, setTraining] = useState(false);
  const [history, setHistory] = useState([]);
  const [status, setStatus] = useState(() => copy.ready);
  const [finalMetrics, setFinalMetrics] = useState(null);
  const [perClassAcc, setPerClassAcc] = useState(null);
  const modelRef = useRef(null);

  const counts = useMemo(() => countByLabel(samples, labels), [samples, labels]);
  const totalSamples = samples.length;
  const minPerClass = labels.reduce((min, label) => Math.min(min, counts[label] ?? 0), Infinity);
  const labelsWithoutData = labels.filter((label) => (counts[label] ?? 0) === 0);
  const canTrain = totalSamples > 0 && labelsWithoutData.length === 0 && minPerClass >= 2 && !training;

  async function importDataset(event) {
    const files = Array.from(event.target.files ?? []);
    if (!files.length) return;

    try {
      const datasets = await Promise.all(files.map(async (file) => {
        const payload = JSON.parse(await file.text());
        return normalizeImportedDataset(payload);
      }));
      const imported = mergeImportedDatasets(datasets);
      setLabels(imported.labels);
      setSamples(imported.samples);
      setHistory([]);
      setFinalMetrics(null);
      setPerClassAcc(null);
      modelRef.current = null;
      window.localStorage.setItem(LABELS_KEY, JSON.stringify(imported.labels));
      window.localStorage.setItem(DATASET_KEY, JSON.stringify(imported.samples));
      setStatus(copy.combined(files.length, imported.samples.length, imported.labels.length));
    } catch (error) {
      setStatus(copy.loadError(error?.message || error));
    } finally {
      event.target.value = '';
    }
  }

  async function train() {
    setTraining(true);
    setHistory([]);
    setFinalMetrics(null);
    setPerClassAcc(null);
    setStatus(copy.preparing);

    let trainXt;
    let trainYt;
    let valXt;
    let valYt;

    try {
      const split = splitDataset(samples, labels);
      trainXt = tf.tensor2d(split.trainX);
      trainYt = tf.tensor1d(split.trainY);
      valXt = tf.tensor2d(split.valX);
      valYt = tf.tensor1d(split.valY);

      const model = tf.sequential();
      model.add(tf.layers.dense({ inputShape: [63], units: 64, activation: 'relu' }));
      model.add(tf.layers.dropout({ rate: 0.2 }));
      model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
      model.add(tf.layers.dropout({ rate: 0.2 }));
      model.add(tf.layers.dense({ units: labels.length, activation: 'softmax' }));
      model.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'sparseCategoricalCrossentropy',
        metrics: ['accuracy']
      });

      const epochs = 80;
      setStatus(copy.trainingStart(epochs));

      await model.fit(trainXt, trainYt, {
        epochs,
        batchSize: 16,
        validationData: [valXt, valYt],
        shuffle: true,
        callbacks: {
          onEpochEnd: async (epoch, logs) => {
            const entry = { epoch: epoch + 1, ...logs };
            setHistory((value) => [...value, entry]);
            const valAcc = logs.val_acc ?? logs.val_accuracy ?? 0;
            setStatus(copy.epoch(epoch + 1, epochs, logs.loss.toFixed(3), (valAcc * 100).toFixed(1)));
            await tf.nextFrame();
          }
        }
      });

      const predLogits = model.predict(valXt);
      const predIdx = await predLogits.argMax(-1).data();
      const correctByClass = Array(labels.length).fill(0);
      const totalByClass = Array(labels.length).fill(0);
      split.valY.forEach((trueIdx, i) => {
        totalByClass[trueIdx] += 1;
        if (predIdx[i] === trueIdx) correctByClass[trueIdx] += 1;
      });
      setPerClassAcc(labels.map((label, idx) => ({
        label,
        correct: correctByClass[idx],
        total: totalByClass[idx],
        acc: totalByClass[idx] > 0 ? correctByClass[idx] / totalByClass[idx] : null
      })));
      predLogits.dispose();

      const finalEval = model.evaluate(valXt, valYt);
      const finalLossArr = await finalEval[0].data();
      const finalAccArr = await finalEval[1].data();
      finalEval.forEach((tensor) => tensor.dispose());
      const finalLoss = finalLossArr[0];
      const finalAcc = finalAccArr[0];
      setFinalMetrics({ loss: finalLoss, acc: finalAcc });

      modelRef.current = model;
      await model.save(`indexeddb://${MODEL_NAME}`);
      window.localStorage.setItem(MODEL_META_KEY, JSON.stringify({
        modelName: MODEL_NAME,
        featureVersion: FEATURE_VERSION,
        labels,
        trainedAt: new Date().toISOString(),
        finalLoss,
        finalAcc
      }));
      setStatus(copy.done((finalAcc * 100).toFixed(1)));
    } catch (error) {
      setStatus(copy.error(error?.message || error));
    } finally {
      trainXt?.dispose();
      trainYt?.dispose();
      valXt?.dispose();
      valYt?.dispose();
      setTraining(false);
    }
  }

  async function downloadModel() {
    if (!modelRef.current) return;
    await exportModelBundle(modelRef.current, {
      modelName: MODEL_NAME,
      featureVersion: FEATURE_VERSION,
      labels,
      trainedAt: new Date().toISOString(),
      finalMetrics
    });
  }

  return (
    <main className="app">
      <AppHeader active="model" language={language} onLanguageChange={selectLanguage} />
      <div className="workspace trainWorkspace">
        <section className="panel">
          <div className="panelHead">
            <div>
              <span>{copy.dataset}</span>
              <h2>{copy.title}</h2>
            </div>
            <BrainCircuit size={26} />
          </div>
          <div className="metricGrid">
            <Metric label={copy.signs} value={labels.length} />
            <Metric label={copy.samples} value={totalSamples} />
            <Metric label={copy.minPerSign} value={Number.isFinite(minPerClass) ? minPerClass : 0} />
          </div>
          <div className="actions">
            <button disabled={training} onClick={() => fileInputRef.current?.click()}>
              <Upload size={17} />
              {copy.upload}
            </button>
            <input
              ref={fileInputRef}
              className="hiddenFileInput"
              type="file"
              accept="application/json,.json"
              multiple
              onChange={importDataset}
            />
          </div>
          {labelsWithoutData.length > 0 && (
            <div className="storageNote">
              <strong>{copy.missingTitle}</strong>
              <span>{copy.missingText(labelsWithoutData.join(', '))}</span>
            </div>
          )}
          <div className="statusBox">
            <span>{status}</span>
          </div>
          <div className="actions">
            <button className="primary" disabled={!canTrain} onClick={train}>
              <BrainCircuit size={17} />
              {training ? copy.training : copy.train}
            </button>
            <button disabled={!modelRef.current || training} onClick={downloadModel}>
              <Download size={17} />
              {copy.download}
            </button>
          </div>
        </section>

        <section className="panel">
          <div className="panelHead">
            <div>
              <span>{copy.progress}</span>
              <h2>{copy.loss}</h2>
            </div>
          </div>
          <LossChart history={history} />
          <div className="metricGrid">
            <Metric label="Val loss" value={finalMetrics ? finalMetrics.loss.toFixed(3) : '—'} />
            <Metric label="Val acc" value={finalMetrics ? `${(finalMetrics.acc * 100).toFixed(1)}%` : '—'} />
            <Metric label="Epocas" value={history.length} />
          </div>
        </section>

        {perClassAcc && (
          <section className="panel perClassPanel">
            <div className="panelHead">
              <div>
                <span>{copy.perSign}</span>
                <h2>{copy.accuracy}</h2>
              </div>
            </div>
            <div className="bars">
              {perClassAcc.map((row) => (
                <div className="barRow" key={row.label}>
                  <span>{row.label}</span>
                  <div><i style={{ width: `${(row.acc ?? 0) * 100}%` }} /></div>
                  <strong>{row.acc != null ? `${(row.acc * 100).toFixed(0)}%` : '—'}</strong>
                </div>
              ))}
            </div>
          </section>
        )}
      </div>
    </main>
  );
}

function UserApp() {
  const tracker = useHandTracker();
  const fileInputRef = useRef(null);
  const modelRef = useRef(null);
  const holdRef = useRef({ label: null, startedAt: 0 });
  const advancingRef = useRef(false);
  const predictionRef = useRef(null);
  const predictingRef = useRef(false);
  const [labels, setLabels] = useState(() => loadStoredJson(MODEL_META_KEY, null)?.labels ?? mergeDefaultLabels(loadStoredJson(LABELS_KEY, DEFAULT_SIGNS)));
  const [modelReady, setModelReady] = useState(false);
  const [status, setStatus] = useState('Modell wird geladen');
  const [prediction, setPrediction] = useState(null);
  const [practiceMode, setPracticeMode] = useState(false);
  const [wordIndex, setWordIndex] = useState(0);
  const [letterIndex, setLetterIndex] = useState(0);
  const [holdProgress, setHoldProgress] = useState(0);
  const [celebrating, setCelebrating] = useState(null);
  const { language, selectLanguage } = usePracticeLanguage();
  const copy = USER_COPY[language] ?? USER_COPY.de;

  const currentWord = PRACTICE_WORDS[wordIndex];
  const targetLetter = currentWord?.[letterIndex] ?? '';
  const completedPart = currentWord.slice(0, letterIndex);
  const remainingPart = currentWord.slice(letterIndex + 1);

  useEffect(() => {
    let cancelled = false;

    async function loadSavedModel() {
      try {
        let model;
        let nextLabels = null;
        try {
          model = await tf.loadLayersModel(`indexeddb://${MODEL_NAME}`);
        } catch {
          const response = await fetch(DEFAULT_MODEL_BUNDLE_URL);
          if (!response.ok) throw new Error('Standardmodell wurde nicht gefunden');
          const bundle = await response.json();
          model = await loadModelBundle(bundle);
          nextLabels = bundle.labels;
          await model.save(`indexeddb://${MODEL_NAME}`);
          window.localStorage.setItem(MODEL_META_KEY, JSON.stringify({
            modelName: bundle.modelName ?? MODEL_NAME,
            featureVersion: bundle.featureVersion ?? FEATURE_VERSION,
            labels: bundle.labels,
            trainedAt: bundle.trainedAt,
            finalMetrics: bundle.finalMetrics
          }));
        }

        if (cancelled) {
          model.dispose();
          return;
        }
        modelRef.current = model;
        if (Array.isArray(nextLabels)) setLabels(nextLabels);
        setModelReady(true);
        setStatus('Modell bereit');
      } catch (error) {
        if (!cancelled) setStatus(`Eigenes Modell hochladen${error?.message ? ` (${error.message})` : ''}`);
      }
    }

    loadSavedModel();
    return () => {
      cancelled = true;
      modelRef.current?.dispose();
    };
  }, []);

  useEffect(() => {
    holdRef.current = { label: null, startedAt: 0 };
    advancingRef.current = false;
    setHoldProgress(0);
  }, [practiceMode, wordIndex, letterIndex]);

  useEffect(() => {
    predictionRef.current = prediction;
  }, [prediction]);

  useEffect(() => {
    if (!modelReady) return undefined;

    let cancelled = false;
    const timer = window.setInterval(async () => {
      if (predictingRef.current) return;
      const currentFeatures = tracker.featuresRef.current;
      if (!modelRef.current || !currentFeatures) {
        predictionRef.current = null;
        setPrediction(null);
        return;
      }

      predictingRef.current = true;
      let input;
      let output;
      try {
        input = tf.tensor2d([currentFeatures]);
        output = modelRef.current.predict(input);
        const values = await output.data();
        if (cancelled) return;

        let bestIndex = 0;
        values.forEach((value, index) => {
          if (value > values[bestIndex]) bestIndex = index;
        });

        const nextPrediction = {
          label: labels[bestIndex] ?? '?',
          confidence: values[bestIndex] ?? 0
        };
        predictionRef.current = nextPrediction;
        setPrediction(nextPrediction);
      } finally {
        input?.dispose();
        output?.dispose();
        predictingRef.current = false;
      }
    }, 100);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [modelReady, tracker.featuresRef, labels]);

  useEffect(() => {
    if (!practiceMode || !targetLetter) return undefined;

    const timer = window.setInterval(() => {
      const currentPrediction = predictionRef.current;
      const isCorrect =
        currentPrediction?.label === targetLetter &&
        currentPrediction.confidence >= 0.45;

      if (!isCorrect) {
        holdRef.current = { label: null, startedAt: 0 };
        setHoldProgress(0);
        return;
      }

      const now = performance.now();
      if (holdRef.current.label !== targetLetter) {
        holdRef.current = { label: targetLetter, startedAt: now };
        setHoldProgress(0);
        return;
      }

      const progress = Math.min(1, (now - holdRef.current.startedAt) / HOLD_TO_CONFIRM_MS);
      setHoldProgress(progress);

      if (progress >= 1 && !advancingRef.current) {
        advancingRef.current = true;
        advancePracticeLetter();
      }
    }, 80);

    return () => window.clearInterval(timer);
  }, [practiceMode, targetLetter, letterIndex, currentWord]);

  function advancePracticeLetter() {
    setHoldProgress(0);
    holdRef.current = { label: null, startedAt: 0 };

    if (letterIndex < currentWord.length - 1) {
      setLetterIndex((value) => value + 1);
      window.setTimeout(() => {
        advancingRef.current = false;
      }, 260);
      return;
    }

    const completedWord = currentWord;
    setStatus(`${copy.wordDone} ${completedWord}`);
    setCelebrating(completedWord);

    if (window.speechSynthesis) {
      window.speechSynthesis.cancel();
      const utter = new SpeechSynthesisUtterance(copy.celebrate(completedWord));
      utter.lang = language === 'de' ? 'de-DE' : 'en-US';
      utter.rate = 1.05;
      window.speechSynthesis.speak(utter);
    }

    window.setTimeout(() => {
      setCelebrating(null);
      setLetterIndex(0);
      setWordIndex((value) => (value + 1) % PRACTICE_WORDS.length);
      advancingRef.current = false;
    }, 2600);
  }

  function nextPracticeWord() {
    setCelebrating(null);
    setLetterIndex(0);
    setWordIndex((value) => (value + 1) % PRACTICE_WORDS.length);
  }

  async function importModel(event) {
    const files = Array.from(event.target.files ?? []);
    if (!files.length) return;

    try {
      if (files.length === 1 && files[0].name.endsWith('.json')) {
        const bundle = JSON.parse(await files[0].text());
        const nextModel = await loadModelBundle(bundle);
        modelRef.current?.dispose();
        modelRef.current = nextModel;
        await nextModel.save(`indexeddb://${MODEL_NAME}`);

        if (Array.isArray(bundle.labels)) {
          setLabels(bundle.labels);
          window.localStorage.setItem(MODEL_META_KEY, JSON.stringify({
            modelName: bundle.modelName ?? MODEL_NAME,
            featureVersion: bundle.featureVersion ?? FEATURE_VERSION,
            labels: bundle.labels,
            trainedAt: bundle.trainedAt,
            finalMetrics: bundle.finalMetrics
          }));
        }

        setModelReady(true);
        setStatus('Modell bereit');
        return;
      }

      const metaFile = files.find((file) => file.name.toLowerCase().includes('meta') && file.name.endsWith('.json'));
      const modelFile = files.find((file) => file.name.endsWith('.json') && !file.name.toLowerCase().includes('meta'));
      const weightFiles = files.filter((file) => file.name.endsWith('.bin'));

      if (!modelFile || weightFiles.length === 0) {
        throw new Error('Bitte Modell-JSON und .bin-Gewichte auswählen.');
      }

      const nextModel = await tf.loadLayersModel(tf.io.browserFiles([modelFile, ...weightFiles]));
      modelRef.current?.dispose();
      modelRef.current = nextModel;
      await nextModel.save(`indexeddb://${MODEL_NAME}`);

      if (metaFile) {
        const meta = JSON.parse(await metaFile.text());
        if (Array.isArray(meta.labels)) {
          setLabels(meta.labels);
          window.localStorage.setItem(MODEL_META_KEY, JSON.stringify(meta));
        }
      }

      setModelReady(true);
      setStatus('Modell bereit');
    } catch (error) {
      setStatus(`Konnte nicht geladen werden: ${error?.message || error}`);
    } finally {
      event.target.value = '';
    }
  }

  function speakPrediction() {
    if (!prediction?.label || !window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(new SpeechSynthesisUtterance(prediction.label));
  }

  const confidence = prediction ? Math.round(prediction.confidence * 100) : 0;

  return (
    <main className="app userApp">
      <header className="userTopbar">
        <a className="userBrand" href="#/app">{APP_NAME}</a>
        <nav>
          <a href="#/app" className="active">{copy.navPractice}</a>
          <a href="#/train">{copy.navData}</a>
          <a href="#/model">{copy.navTraining}</a>
          <a href="#/info">{copy.navStory}</a>
        </nav>
        <LanguageControl language={language} onChange={selectLanguage} />
      </header>

      <div className="userWorkspace">
        <section className="userCameraCard">
          <div className="userVideoShell">
            <video ref={tracker.videoRef} muted playsInline />
            <HandFrame landmarks={tracker.landmarks} />
            <div className="cameraPredictionOverlay">
              <strong>{prediction?.label ?? '-'}</strong>
              <span>{prediction ? `${confidence}%` : modelReady ? copy.noHand : copy.loading}</span>
            </div>
            {celebrating && (
              <div className="celebrationOverlay" role="status" aria-live="polite">
                <div className="celebrationCard">
                  <span className="celebrationEyebrow">{copy.wordDone}</span>
                  <strong>{celebrating}</strong>
                  <span className="celebrationSubtitle">{copy.wordDoneSubtitle(celebrating)}</span>
                </div>
              </div>
            )}
          </div>
          <div className="userCameraStatus">
            <span>{tracker.features ? copy.handDetected : tracker.status}</span>
            <strong>{modelReady ? copy.modelActive : copy.noModel}</strong>
          </div>
        </section>

        <section className="userResult">
          <span>{practiceMode ? copy.practice : copy.detected}</span>
          {practiceMode ? (
            <>
              <div className="practiceWord" aria-label={`Wort ${currentWord}`}>
                <span>{completedPart}</span>
                <strong>{targetLetter}</strong>
                <span>{remainingPart}</span>
              </div>
              <div className="practiceHelp">
                <img src={`${LETTER_IMAGE_BASE}/${targetLetter}.png`} alt={`Help for ${targetLetter}`} />
              </div>
            </>
          ) : (
            <strong>{prediction?.label ?? '-'}</strong>
          )}
          <div className="confidenceLine">
            <i style={{ width: `${practiceMode ? holdProgress * 100 : confidence}%` }} />
          </div>
          <small>
            {practiceMode
              ? `${letterIndex + 1}/${currentWord.length} - ${prediction?.label ?? '-'}`
              : prediction ? `${confidence}% ${copy.confidence}` : status}
          </small>

          <div className="userActions">
            <button className={practiceMode ? 'primary active' : 'primary'} onClick={() => setPracticeMode((value) => !value)}>
              <Image size={17} />
              {practiceMode ? copy.free : copy.practiceButton}
            </button>
            <button disabled={!prediction} onClick={speakPrediction}>
              <Volume2 size={17} />
              {copy.speak}
            </button>
            {practiceMode && (
              <button onClick={nextPracticeWord}>
                <RotateCcw size={17} />
                {copy.newWord}
              </button>
            )}
            <button onClick={() => fileInputRef.current?.click()}>
              <Upload size={17} />
              {copy.ownModel}
            </button>
            <input
              ref={fileInputRef}
              className="hiddenFileInput"
              type="file"
              accept="application/json,.json,.bin"
              multiple
              onChange={importModel}
            />
          </div>
        </section>
      </div>
      <section className="projectHistory">
        <div>
          <span>{copy.historyEyebrow}</span>
          <h2>{copy.historyTitle}</h2>
          <p>
            {copy.historyText}
            <strong> {copy.ownModel}</strong>.
          </p>
        </div>
        <div className="historySteps">
          <a href="#/train">
            <strong>{copy.stepDataTitle}</strong>
            <span>{copy.stepDataText}</span>
          </a>
          <a href="#/model">
            <strong>{copy.stepTrainTitle}</strong>
            <span>{copy.stepTrainText}</span>
          </a>
          <a href="#/app">
            <strong>{copy.stepPracticeTitle}</strong>
            <span>{copy.stepPracticeText}</span>
          </a>
        </div>
      </section>
    </main>
  );
}

const STORY_COPY = {
  de: {
    navPractice: 'Üben',
    navData: 'Daten',
    navTraining: 'Training',
    navStory: 'Story',
    eyebrow: 'Engineering Story',
    title: 'Wie ich dem Browser beigebracht habe, Hände zu lesen',
    intro: 'Echtzeit-Gebärdenerkennung im Browser klingt geradeaus — bis ich es bauen wollte. Drei Versuche, ein paar verlorene Wochenenden und eine entscheidende Einsicht später lief es dann doch.',
    attempt1Tag: 'Versuch 01',
    attempt1Title: 'Vorgefertigte öffentliche Modelle',
    attempt1Body: 'Erster Reflex: ein fertiges Sign-Language-Modell von HuggingFace oder TF Hub nehmen. Hat nicht funktioniert. Die wenigen Modelle sind auf ASL trainiert — nicht auf DGS — und brechen bei Beleuchtung, Kamerawinkel oder Hauttönen außerhalb ihres Datensatzes zusammen. Selbst mit perfekter Kameraführung blieb ich bei ~30% Genauigkeit.',
    attempt2Tag: 'Versuch 02',
    attempt2Title: 'Eigenes CNN auf Bildern',
    attempt2Body: 'Logischer nächster Schritt: eigene Bilder, eigenes Modell. Ich habe ein paar hundert Frames pro Buchstabe gesammelt und MobileNetV2 fine-getunt. In meinem Wohnzimmer ~85%. Auf einem anderen Rechner, mit anderem Hintergrund — Absturz auf 40%. Das Modell hatte die Tapete gelernt, nicht die Hand.',
    attempt3Tag: 'Lösung',
    attempt3Title: 'Hand-Landmarks, relativ zur Handfläche',
    attempt3Body: 'Die Einsicht: ich brauche kein Bild, sondern die Geometrie der Hand. MediaPipe Hand Landmarker liefert 21 3D-Punkte pro Hand, robust gegen Beleuchtung, Hintergrund und Hauttöne. Ich normalisiere die Punkte relativ zum Handgelenk und skaliere auf die Bounding-Box — Position, Größe und Distanz spielen keine Rolle mehr. Der Eingang schrumpft von 224×224×3 (≈150k Werte) auf 63 Zahlen. Ein MLP mit 2 Hidden Layers reicht, klassifiziert auf CPU in unter 1 ms und läuft komplett im Browser.',
    impactTag: 'Wirkung',
    impactTitle: 'Warum das ein guter Hack ist',
    impactBody: 'Ich habe das Problem von „mehr Daten sammeln" auf „bessere Features bauen" verlagert. Statt zehntausender Fotos: ~80 Aufnahmen pro Zeichen. Statt schwerem CNN: ein MLP mit 12 KB. Statt Cloud-Inferenz: alles im Browser, ohne Server, ohne Latenz, ohne Datenschutz-Risiko. Die richtige Repräsentation schlägt das größere Modell.',
    stackTag: 'Stack',
    stackTitle: 'Unter der Haube',
    stackItems: [
      ['MediaPipe Hand Landmarker', '21 3D-Punkte, auf Millionen Bildern vortrainiert.'],
      ['Custom Normalisierung', 'Punkte relativ zum Handgelenk, skaliert auf Bounding-Box.'],
      ['TensorFlow.js MLP', '2 Hidden Layers, im Browser trainiert, in IndexedDB gespeichert.'],
      ['Zero Backend', 'Inferenz, Training und Export komplett client-seitig.']
    ],
    metricsTitle: 'In Zahlen',
    metrics: [
      ['~80', 'Aufnahmen pro Zeichen'],
      ['63', 'Eingangs-Features'],
      ['12 KB', 'Modellgröße'],
      ['<1 ms', 'Inferenzzeit'],
      ['100%', 'Client-Side']
    ],
    backCta: 'Zur Übung'
  },
  en: {
    navPractice: 'Practice',
    navData: 'Data',
    navTraining: 'Training',
    navStory: 'Story',
    eyebrow: 'Engineering Story',
    title: 'How I taught a browser to read hands',
    intro: 'Real-time sign recognition in the browser sounds straightforward — until I tried to build it. Three attempts, a few lost weekends, and one crucial insight later, it works.',
    attempt1Tag: 'Attempt 01',
    attempt1Title: 'Pretrained public models',
    attempt1Body: 'First reflex: grab a sign-language model from HuggingFace or TF Hub. Did not work. The few public models are trained on ASL — not DGS — and collapse the moment lighting, camera angle, or skin tone drift outside their training set. Even with perfect framing, accuracy sat around 30%.',
    attempt2Tag: 'Attempt 02',
    attempt2Title: 'My own image-classification CNN',
    attempt2Body: 'Logical next step: my own images, my own model. I collected a few hundred frames per letter and fine-tuned MobileNetV2. In my living room: ~85%. On another machine, different background — crashed to 40%. The model had learned the wallpaper, not the hand.',
    attempt3Tag: 'Solution',
    attempt3Title: 'Hand landmarks, relative to the palm',
    attempt3Body: 'The insight: I do not need an image — I need the geometry of the hand. MediaPipe Hand Landmarker returns 21 3D points per hand, robust to lighting, background, and skin tone. I normalize each point relative to the wrist and scale by the bounding box — position, size, and distance stop mattering. The input shrinks from 224×224×3 (≈150k values) to 63 numbers. An MLP with 2 hidden layers is enough, classifies in under 1 ms on CPU, and runs entirely in the browser.',
    impactTag: 'Impact',
    impactTitle: 'Why this is a good hack',
    impactBody: 'I moved the problem from "collect more data" to "engineer better features." Instead of tens of thousands of photos: ~80 captures per sign. Instead of a heavy CNN: an MLP with 12 KB. Instead of cloud inference: everything in the browser — no servers, no latency, no privacy risk. The right representation beats the bigger model.',
    stackTag: 'Stack',
    stackTitle: 'Under the hood',
    stackItems: [
      ['MediaPipe Hand Landmarker', '21 3D points, pretrained on millions of images.'],
      ['Custom normalization', 'Points relative to the wrist, scaled to the bounding box.'],
      ['TensorFlow.js MLP', '2 hidden layers, trained in the browser, stored in IndexedDB.'],
      ['Zero backend', 'Inference, training, and export run entirely client-side.']
    ],
    metricsTitle: 'By the numbers',
    metrics: [
      ['~80', 'captures per sign'],
      ['63', 'input features'],
      ['12 KB', 'model size'],
      ['<1 ms', 'inference time'],
      ['100%', 'client-side']
    ],
    backCta: 'Back to practice'
  }
};

function InfoApp() {
  const { language, selectLanguage } = usePracticeLanguage();
  const copy = STORY_COPY[language] ?? STORY_COPY.de;

  return (
    <main className="app userApp storyApp">
      <header className="userTopbar">
        <a className="userBrand" href="#/app">{APP_NAME}</a>
        <nav>
          <a href="#/app">{copy.navPractice}</a>
          <a href="#/train">{copy.navData}</a>
          <a href="#/model">{copy.navTraining}</a>
          <a href="#/info" className="active">{copy.navStory}</a>
        </nav>
        <LanguageControl language={language} onChange={selectLanguage} />
      </header>

      <section className="storyHero">
        <span className="storyEyebrow">{copy.eyebrow}</span>
        <h1>{copy.title}</h1>
        <p>{copy.intro}</p>
      </section>

      <section className="storySteps">
        <article className="storyStep storyStepFailed">
          <span className="storyStepTag">{copy.attempt1Tag}</span>
          <h2>{copy.attempt1Title}</h2>
          <p>{copy.attempt1Body}</p>
        </article>

        <article className="storyStep storyStepFailed">
          <span className="storyStepTag">{copy.attempt2Tag}</span>
          <h2>{copy.attempt2Title}</h2>
          <p>{copy.attempt2Body}</p>
        </article>

        <article className="storyStep storyStepWin">
          <span className="storyStepTag">{copy.attempt3Tag}</span>
          <h2>{copy.attempt3Title}</h2>
          <p>{copy.attempt3Body}</p>
        </article>
      </section>

      <section className="storySplit">
        <div className="storyImpact">
          <span className="storyEyebrow">{copy.impactTag}</span>
          <h2>{copy.impactTitle}</h2>
          <p>{copy.impactBody}</p>
        </div>

        <div className="storyMetrics">
          <span className="storyEyebrow">{copy.metricsTitle}</span>
          <div className="storyMetricsGrid">
            {copy.metrics.map(([value, label]) => (
              <div key={label} className="storyMetric">
                <strong>{value}</strong>
                <span>{label}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="storyStack">
        <span className="storyEyebrow">{copy.stackTag}</span>
        <h2>{copy.stackTitle}</h2>
        <div className="storyStackGrid">
          {copy.stackItems.map(([title, body]) => (
            <div key={title} className="storyStackItem">
              <strong>{title}</strong>
              <span>{body}</span>
            </div>
          ))}
        </div>
      </section>

      <section className="storyCta">
        <a href="#/app">{copy.backCta} →</a>
      </section>
    </main>
  );
}

function AppRouter() {
  const hash = useHashRoute();
  if (hash.startsWith('#/app')) return <UserApp />;
  if (hash.startsWith('#/model')) return <TrainModelApp />;
  if (hash.startsWith('#/info')) return <InfoApp />;
  return <TrainerApp />;
}

createRoot(document.getElementById('root')).render(<AppRouter />);
