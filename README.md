# DGS Fingeralphabet Trainer

Web-App zum Üben des deutschen Fingeralphabets mit Webcam und lokalem TensorFlow.js-Modell.

## Ansichten

- `#/app`: Hauptansicht zum Üben.
- `#/train`: lokale Datenerfassung als JSON.
- `#/model`: Training mit einem oder mehreren JSON-Datensätzen.

## Projektablauf

1. In `#/train` Handzeichen aufnehmen und den Datensatz exportieren.
2. In `#/model` mehrere JSON-Dateien kombinieren und ein Modell trainieren.
3. Das Modell als Bundle herunterladen.
4. In `#/app` das Standardmodell verwenden oder ein eigenes Bundle hochladen.

Die App speichert keine Fotos und kein Video. Pro Aufnahme werden nur 63 normalisierte Landmark-Werte gespeichert.

## Eigenes Modell

Wer eigene Daten verwenden möchte, kann in der Trainingsansicht ein neues Modell trainieren und anschließend in der Übungsansicht über `Eigenes Modell` hochladen.

## Lokale Entwicklung

```bash
npm install
npm run dev
```

## Build

```bash
npm run build
```
