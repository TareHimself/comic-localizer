# Comic Localizer

## What
Extensible comic localizer

## Install
- This repo uses [uv](https://github.com/astral-sh/uv) for package management, see individual parts below for specifics
  
## Parts
- [Actual Translator](./translator/README.md)
- A [UI](./ui/README.md) for translating individual images
- A basic [CLI](./cli/README.md) example
  
## Limitations
- Yolo by ultralytics is AGPL, need to find alternative detector and segmenter (or maybe remove from repo so repo is not AGPL)
- Currently no form of text color detection so text will always be black
- Only horizontal left to right text layout is supported for output
- Very large vertical images are not split so either they work or they dont
- Dataset is only japanese text so detection might not work on other languages
- Aside from the yolo models, all other models are general-purpose and may or may not work well on manga

## Models Included
- Yolo detection and segmentation
- deepfillv2 inpainting
- lama inpainting

<!--
Links are currently down, and I'm not sure when I will be able to bring them back up
## Datasets

### Detection

<a href="https://universe.roboflow.com/tarehimself/manga-translator-detection">
    <img src="https://app.roboflow.com/images/download-dataset-badge.svg"></img>
</a>

### Segmentation

<a href="https://universe.roboflow.com/tarehimself/manga-translator-segmentation">
    <img src="https://app.roboflow.com/images/download-dataset-badge.svg"></img>
</a>
-->
<!-- 
## Glossary

- Bubble: a speech bubble
- Free text: text found on pages but not in speech bubbles
- Bubble Text: text within speech bubbles -->
