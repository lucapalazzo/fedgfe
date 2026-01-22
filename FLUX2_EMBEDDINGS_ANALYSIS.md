# FLUX2-Klein: Analisi Supporto Embeddings T5/CLIP

## Domanda
Posso passare gli embedding testuali T5 e CLIP direttamente al modello `x/flux2-klein:latest` tramite Ollama?

## Risposta Breve
**NO** tramite Ollama API, **SÌ** tramite Python/Diffusers direttamente.

---

## Analisi Dettagliata

### 1. FLUX2-Klein Architecture

**FLUX.2 [klein]** (rilasciato gennaio 2026):
- Modello **9B** con text embedder **Qwen3 8B** (NON più CLIP+T5!)
- Step-distilled a **4 inference steps**
- Generazione **sub-second** su GPU consumer
- **Cambio architetturale importante**: usa Qwen3 invece di CLIP+T5

Fonte: [FLUX.2 [klein] Blog](https://bfl.ai/blog/flux2-klein-towards-interactive-visual-intelligence)

### 2. FLUX.1 vs FLUX.2 Text Encoders

#### FLUX.1 (architettura precedente):
- **CLIP**: clip-vit-large-patch14 (768 dim)
- **T5**: google/t5-v1_1-xxl (4096 dim)
- Dual encoder approach
- 77 token limit per CLIP, 512 per T5

#### FLUX.2 [klein]:
- **Qwen3 8B** text embedder
- Architettura unificata (non più dual encoder)

Fonte: [HuggingFace Flux Docs](https://huggingface.co/docs/diffusers/main/api/pipelines/flux)

### 3. Ollama API Limitations

**Test eseguiti**: ❌ Tutti falliti con error 500

```json
// Test effettuati:
{
  "model": "x/flux2-klein:latest",
  "embeddings": [...],           // ❌ Non supportato
  "t5_embeddings": [...],        // ❌ Non supportato
  "clip_embeddings": [...],      // ❌ Non supportato
  "encoder_hidden_states": [...] // ❌ Non supportato
}
```

**Motivi**:
1. Ollama template è semplicemente `{{ .Prompt }}` (solo testo)
2. `/api/generate` non espone parametri low-level per embeddings
3. `/api/embeddings` serve per **generare** embeddings, non per passarli come input
4. FLUX2-klein in Ollama ha problemi con image runner (exit 255)

Fonte: [Ollama Embeddings Docs](https://docs.ollama.com/capabilities/embeddings)

### 4. Soluzioni Alternative

#### Opzione 1: Python Diffusers (CONSIGLIATO per il tuo caso)

```python
from diffusers import FluxPipeline
import torch

# Carica pipeline
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B",
    torch_dtype=torch.bfloat16
)
pipe.to("cuda")

# IMPORTANTE: Per FLUX.1 puoi passare embeddings custom
# Per FLUX.2-klein devi verificare se supporta ancora questa feature

# Esempio con FLUX.1 (dual encoder):
prompt_embeds = your_custom_t5_embeddings  # [batch, seq_len, 4096]
pooled_prompt_embeds = your_custom_clip_embeddings  # [batch, 768]

image = pipe(
    prompt_embeds=prompt_embeds,
    pooled_prompt_embeds=pooled_prompt_embeds,
    num_inference_steps=4,
    guidance_scale=3.5
).images[0]
```

**Note**:
- FLUX.1 supporta `prompt_embeds` e `pooled_prompt_embeds`
- FLUX.2-klein potrebbe avere API diversa (Qwen3-based)
- Verifica documentazione aggiornata per FLUX.2

Fonte: [Diffusers FLUX Pipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux.py)

#### Opzione 2: Black Forest Labs API

```python
# Official FLUX.2 inference repo
# https://github.com/black-forest-labs/flux2
```

#### Opzione 3: Convertire Embeddings → Text (Workaround)

Se gli embeddings provengono da adapter:
1. Passa gli embeddings attraverso un decoder/mapper
2. Genera testo approssimativo
3. Usa il testo con Ollama

**Limitazione**: Perdita di informazione nella conversione

---

## Raccomandazione per Federated Learning

### Il tuo scenario:
- Hai **adapter T5/CLIP** trainati sui nodi
- Vuoi generare immagini usando gli embeddings degli adapter
- Hai generatori VAE per classe

### Soluzione migliore:

#### 1. Usa FLUX direttamente con Diffusers (non Ollama)

```python
from diffusers import FluxPipeline
import torch

class AdapterToFluxBridge:
    def __init__(self):
        self.flux_pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",  # O FLUX.2 se supporta embeddings
            torch_dtype=torch.bfloat16
        )
        self.flux_pipe.to("cuda")

    def generate_from_adapter_outputs(self, t5_embedding, clip_embedding):
        """
        Genera immagine da output adapter T5/CLIP

        Args:
            t5_embedding: torch.Tensor [seq_len, 4096] from adapter.t5(audio_features)
            clip_embedding: torch.Tensor [768] from adapter.clip(audio_features)
        """
        # Reshape per batch
        prompt_embeds = t5_embedding.unsqueeze(0)  # [1, seq_len, 4096]
        pooled_prompt_embeds = clip_embedding.unsqueeze(0)  # [1, 768]

        # Genera
        image = self.flux_pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            num_inference_steps=20,
            guidance_scale=3.5
        ).images[0]

        return image
```

#### 2. Oppure: Chain VAE → Text Decoder → FLUX

```python
# 1. Genera samples con VAE
synthetic_audio = vae_generator.sample(num_samples=1)

# 2. Passa attraverso adapter
t5_embed = adapter.t5(synthetic_audio)
clip_embed = adapter.clip(synthetic_audio)

# 3. (Opzionale) Decodifica a testo
text_approx = t5_decoder(t5_embed)  # Serve un decoder addestrato

# 4. Genera con FLUX via Ollama
ollama.generate(model="flux", prompt=text_approx)
```

---

## Conclusioni

### ✅ Possibile:
- Usare embeddings T5/CLIP custom con **FLUX.1** via **Diffusers**
- Bypass completo di Ollama
- Controllo diretto su tutto il pipeline

### ❌ Non Possibile:
- Passare embeddings via **Ollama API**
- FLUX.2-klein in Ollama ha problemi (exit 255)
- Perdita architetturale: FLUX.2 usa Qwen3 (no CLIP+T5)

### 📌 Raccomandazione Finale:

Per il tuo progetto federated learning:
1. **Usa FLUX.1-dev con Diffusers** (supporta CLIP+T5 embeddings)
2. **Integra direttamente** gli output degli adapter
3. **Evita Ollama** per questo caso d'uso specifico
4. Se necessario, crea un **microservizio Python** che espone API REST per generazione da embeddings

---

## Fonti

- [FLUX.2 Klein Blog](https://bfl.ai/blog/flux2-klein-towards-interactive-visual-intelligence)
- [HuggingFace Flux Docs](https://huggingface.co/docs/diffusers/main/api/pipelines/flux)
- [Diffusers FLUX Pipeline Code](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux.py)
- [Ollama Embeddings Documentation](https://docs.ollama.com/capabilities/embeddings)
- [Black Forest Labs FLUX.2 Repo](https://github.com/black-forest-labs/flux2)
- [Diffusers FLUX.2 Blog](https://huggingface.co/blog/flux-2)

---

## Prossimi Passi

Se vuoi procedere con l'integrazione:
1. Installa `diffusers>=0.30.0` (supporta FLUX.2)
2. Scarica FLUX.1-dev (se serve CLIP+T5) o FLUX.2-klein (più veloce)
3. Testa passaggio embeddings adapter → FLUX
4. Valuta trade-off: qualità vs velocità (FLUX.1 vs FLUX.2)
