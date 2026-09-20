# Stable Diffusion – PyTorch Implementation

Een eigen implementatie van belangrijke onderdelen van **Stable Diffusion** met **Python en PyTorch**.

Dit project is gemaakt om inzicht te krijgen in de interne werking van Stable Diffusion door de belangrijkste onderdelen zelf te implementeren, in plaats van uitsluitend gebruik te maken van bestaande high-level libraries.

Het project is geïnspireerd door en nagebouwd aan de hand van het volgende open-source project:

* https://github.com/hkproj/pytorch-stable-diffusion

Het doel van dit project is voornamelijk **leren, experimenteren en begrijpen hoe Stable Diffusion technisch is opgebouwd**.

---

## 📌 Project

Stable Diffusion is een generatief deep-learningmodel waarmee afbeeldingen kunnen worden gegenereerd op basis van tekstprompts.

In grote lijnen bestaat het proces uit:

```text
Text Prompt
    │
    ▼
CLIP / Text Encoder
    │
    ▼
Text Embeddings
    │
    ▼
Diffusion Model / U-Net
    │
    ▼
Latent Representation
    │
    ▼
VAE Decoder
    │
    ▼
Generated Image
```

Deze repository probeert deze onderdelen zo veel mogelijk zelf te implementeren met PyTorch.

---

## 🧠 Belangrijkste onderdelen

Het project bevat verschillende onderdelen van de Stable Diffusion-pipeline.

### `attention.py`

Implementatie van attention-mechanismen die worden gebruikt binnen het diffusion-model.

Onder andere:

* Self-Attention
* Cross-Attention
* Multi-Head Attention

Attention zorgt ervoor dat het model relaties kan leggen tussen verschillende delen van de input.

---

### `clip.py`

Onderdeel voor het verwerken van tekst met CLIP.

De tekstprompt wordt hierbij omgezet naar numerieke representaties die door het diffusion-model gebruikt kunnen worden.

Globaal:

```text
"An astronaut riding a horse"
            │
            ▼
       Tokenization
            │
            ▼
       CLIP Encoder
            │
            ▼
      Text Embeddings
```

---

### `encoder.py`

De VAE encoder.

Deze zet een afbeelding om van de oorspronkelijke pixelruimte naar een compacte **latent representation**.

```text
Image
  │
  ▼
VAE Encoder
  │
  ▼
Latent Space
```

Door in de latent space te werken hoeft het diffusion-model niet rechtstreeks met alle pixels te werken.

---

### `decoder.py`

De tegenhanger van de encoder.

De decoder zet de latent representation weer om naar een afbeelding.

```text
Latent Space
     │
     ▼
VAE Decoder
     │
     ▼
Image
```

---

### `diffusion.py`

Bevat de belangrijkste onderdelen van het diffusion-model.

Het model probeert tijdens de generatie stapsgewijs ruis uit een latent representation te verwijderen.

Conceptueel:

```text
Random Noise
     │
     ▼
Denoising Step
     │
     ▼
Less Noise
     │
     ▼
Denoising Step
     │
     ▼
Less Noise
     │
     ▼
...
     │
     ▼
Generated Image
```

---

### `ddpm.py`

Bevat onderdelen die betrekking hebben op **Denoising Diffusion Probabilistic Models (DDPM)**.

Een diffusion-model maakt gebruik van een proces waarbij tijdens training ruis aan data wordt toegevoegd.

Tijdens generatie wordt dit proces omgekeerd.

```text
Forward Process

Image
  ↓
+ Noise
  ↓
More Noise
  ↓
Almost Random Noise
```

En vervolgens:

```text
Reverse Process

Random Noise
  ↓
Denoising
  ↓
Denoising
  ↓
Denoising
  ↓
Generated Image
```

---

### `pipeline.py`

De pipeline combineert de verschillende onderdelen tot één geheel.

Globaal wordt de volgende keten uitgevoerd:

```text
Prompt
  │
  ▼
Tokenizer
  │
  ▼
CLIP
  │
  ▼
Text Embeddings
  │
  ▼
Diffusion / U-Net
  │
  ▼
Latent Image
  │
  ▼
VAE Decoder
  │
  ▼
Image
```

---

### `model_loader.py`

Laadt de benodigde modelgewichten en andere modeldata.

Hiermee kunnen vooraf getrainde Stable Diffusion-gewichten worden ingelezen.

---

### `model_converter.py`

Bevat functionaliteit voor het converteren van modelgewichten naar het formaat dat door de eigen PyTorch-implementatie wordt gebruikt.

Dit is vooral relevant wanneer bestaande Stable Diffusion-checkpoints worden gebruikt.

---

## 📓 Notebooks

### `add_noise.ipynb`

Notebook om het toevoegen van ruis binnen het diffusion-proces te onderzoeken.

Dit notebook is vooral bedoeld om experimenteel inzicht te krijgen in het diffusion-proces.

### `demo.ipynb`

Eenvoudige demonstratie van de implementatie.

---

## 📁 Projectstructuur

De belangrijkste structuur van het project is:

```text
.
├── sd/
│   ├── attention.py
│   ├── clip.py
│   ├── ddpm.py
│   ├── decoder.py
│   ├── diffusion.py
│   ├── encoder.py
│   ├── model_converter.py
│   ├── model_loader.py
│   ├── pipeline.py
│   │
│   ├── add_noise.ipynb
│   └── demo.ipynb
│
├── data/
│   ├── vocab.json
│   ├── merges.txt
│   └── *.ckpt
│
└── README.md
```

---

# ⚙️ Installatie

## 1. Repository clonen

```bash
git clone <jouw-repository-url>
cd <jouw-repository>
```

---

## 2. Python environment maken

Het wordt aanbevolen om een aparte virtual environment te gebruiken.

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Dependencies installeren

Installeer PyTorch en de benodigde Python-packages.

Bijvoorbeeld:

```bash
pip install torch torchvision
```

Eventueel kunnen aanvullende dependencies worden geïnstalleerd afhankelijk van de gebruikte onderdelen van het project.

---

# 📦 Modelbestanden

Voor het uitvoeren van Stable Diffusion zijn modelgewichten en tokenizerbestanden nodig.

De oorspronkelijke Stable Diffusion v1.5 release bevat onder andere:

```text
vocab.json
merges.txt
v1-5-pruned-emaonly.ckpt
```

Deze bestanden horen in:

```text
data/
```

De grote modelbestanden worden **niet** in deze repository opgeslagen.

---

# 🚀 Gebruik

Na installatie kunnen de onderdelen vanuit Python worden gebruikt.

Een eenvoudige pipeline ziet conceptueel als volgt uit:

```python
from sd.pipeline import generate

image = generate(
    prompt="A futuristic city at night"
)
```

De exacte aanroep kan afhankelijk zijn van de huidige implementatie van `pipeline.py`.

Voor experimenten kan ook:

```text
demo.ipynb
```

worden geopend met Jupyter Notebook of JupyterLab.

---

# 🔬 Hoe Stable Diffusion werkt

Het belangrijkste idee achter Stable Diffusion is dat het model niet rechtstreeks afbeeldingen genereert vanuit niets.

In plaats daarvan wordt gewerkt in een **latent space**.

Een vereenvoudigd proces:

### 1. Tekst verwerken

```text
Text Prompt
     ↓
Tokenizer
     ↓
CLIP
     ↓
Text Embedding
```

### 2. Starten met ruis

```text
Random Latent Noise
```

### 3. Iteratief denoisen

Het diffusion-model gebruikt de tekstembeddings om te voorspellen hoe de ruis moet worden verwijderd.

```text
Noise
 ↓
Denoise
 ↓
Denoise
 ↓
Denoise
 ↓
...
 ↓
Latent Image
```

### 4. Latent naar afbeelding

De VAE decoder zet de uiteindelijke latent representation om naar pixels.

```text
Latent
  ↓
VAE Decoder
  ↓
RGB Image
```

---

# 🎯 Doel van dit project

Dit project is voornamelijk bedoeld als **educatief en experimenteel project**.

Het doel is om beter te begrijpen hoe moderne generatieve AI-systemen zijn opgebouwd.

Daarbij wordt onder andere gekeken naar:

* PyTorch
* Neural Networks
* Attention
* Transformers
* CLIP
* VAE
* Latent Space
* Diffusion Models
* DDPM
* U-Net architecturen
* Text-to-Image generatie
* Model checkpoints
* Tensor operations

Door de onderdelen zelf te implementeren wordt geprobeerd niet alleen Stable Diffusion te gebruiken, maar ook te begrijpen **wat er onder de motorkap gebeurt**.

---

# 🧪 Experimenten

Mogelijke experimenten binnen dit project:

* verschillende prompts testen
* verschillende diffusion steps gebruiken
* verschillende guidance scales onderzoeken
* noise schedules vergelijken
* latent representations bekijken
* attention onderzoeken
* CLIP embeddings analyseren
* verschillende Stable Diffusion checkpoints testen
* CPU- en GPU-prestaties vergelijken

---

# 📚 Referentie

Dit project is geïnspireerd door:

**HKproj – PyTorch Stable Diffusion**

https://github.com/hkproj/pytorch-stable-diffusion

De implementatie in deze repository is bedoeld als een eigen leer- en experimenteerproject.

Ook zijn de volgende projecten relevant voor het begrijpen van Stable Diffusion:

* CompVis Stable Diffusion
* Hugging Face Diffusers
* andere open-source PyTorch Stable Diffusion implementaties

---

# ⚠️ Disclaimer

Dit project is niet bedoeld als een volledige vervanging voor professionele Stable Diffusion-frameworks zoals Hugging Face Diffusers.

De implementatie is voornamelijk gericht op:

```text
Learning
   ↓
Understanding
   ↓
Experimenting
   ↓
Building from scratch
```

De code kan daarom eenvoudiger zijn dan productiegerichte implementaties.

---

# 📄 License

Controleer de licenties van de gebruikte modelgewichten, tokenizerbestanden en eventuele externe code voordat deze voor commerciële doeleinden worden gebruikt.

De modelgewichten van Stable Diffusion kunnen onder andere hun eigen licentievoorwaarden hebben.

---

## 👨‍💻 Projectstatus

Dit project is **in ontwikkeling**.

Nieuwe onderdelen kunnen worden toegevoegd naarmate de implementatie verder wordt uitgebreid en verbeterd.
