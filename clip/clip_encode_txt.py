import torch
import clip
import numpy as np


def extract_clip_text_features(text, output_path):
    # 1. Carga del modelo con parámetros originales
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device, jit=False)
    model = model.float()  # Forzar precisión FP32

    # 2. Tokenización oficial de CLIP (left-truncate + padding)
    text_tokens = clip.tokenize([text], truncate=True).to(device)

    # 3. Extracción de características
    with torch.no_grad():
        # 3a. Embeddings crudos
        x = model.token_embedding(text_tokens).type(model.dtype)

        # 3b. Posicionales exactas
        x += model.positional_embedding.type(model.dtype)

        # 3c. Transformer con parámetros originales
        x = x.permute(1, 0, 2)  # [NLD -> LND]
        x = model.transformer(x)
        x = x.permute(1, 0, 2)  # [LND -> NLD]

        # 3d. Capa de normalización final
        x = model.ln_final(x).float()

        # 3e. Proyección y normalización
        text_features = x[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)] @ model.text_projection

        # MODIFICACIÓN AQUÍ: Multiplicar por el factor de escala ~10.6 antes de normalizar
        # O mejor aún, NO normalizar si el original no lo hace
        # text_features = text_features * 10.6  # Opción 1: Aplicar factor de escala
        # text_features = text_features / text_features.norm(dim=1, keepdim=True)  # Comentar esta línea si el original no normaliza

    # 4. Formateo compatible
    last_hidden_state = x[0].cpu().numpy()[:77]  # [77,512]
    pooler_output = text_features[0].cpu().numpy()  # [512]

    # 5. Recorte al contexto real del texto
    eos_pos = (text_tokens[0] == 49407).nonzero(as_tuple=True)[0][0].item()
    last_hidden_state = last_hidden_state[:eos_pos + 1]

    np.savez(
        output_path,
        last_hidden_state=last_hidden_state.astype(np.float32),
        pooler_output=pooler_output.astype(np.float32)
    )


# # Test de validación
# text = "some military patriots takes us through their safety procedures and measures."
# extract_clip_text_features(text, "my_features.npz")
#
# # Análisis comparativo
# original = np.load(r"D:\Downloads\qvhighlight\qvhighlight\clip_text_features\qid9769.npz")
#
# nuevo = np.load("my_features.npz")
#
# print("=== Pooler Output ===")
# print("Original:", original['pooler_output'][:5].round(4))
# print("Nuevo:   ", nuevo['pooler_output'][:5].round(4))
#
# # Calculamos la proporción entre los vectores
# ratio = original['pooler_output'] / (nuevo['pooler_output'] + 1e-10)  # Evitamos división por cero
# print("\n=== Ratio entre vectores (promedio) ===")
# print(f"Ratio promedio: {ratio.mean():.4f}")
#
# print("\n=== Hidden States (Norma L2) ===")
# print("Original:", np.linalg.norm(original['last_hidden_state'], axis=1).round(2))
# print("Nuevo:   ", np.linalg.norm(nuevo['last_hidden_state'], axis=1).round(2))
#
# # Verificamos si los vectores son paralelos (coseno similar a 1)
# cosine_sim = np.dot(original['pooler_output'], nuevo['pooler_output']) / (
#         np.linalg.norm(original['pooler_output']) * np.linalg.norm(nuevo['pooler_output']))
# print(f"\nSimilitud coseno entre vectores: {cosine_sim:.6f}")