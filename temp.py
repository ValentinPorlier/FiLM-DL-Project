import os
import random
from PIL import Image

# --- CONFIGURATION (Vérifie bien ces chemins) ---
# Dossier source (Ton HDD avec des tonnes d'images)
source_root = r"D:/00/archive" 

# Dossier destination (Ton SSD pour le training rapide)
# On ajoute /images à la fin pour que ImageFolder fonctionne direct
dest_folder = r"D:/00/img_style_resized" 


styles_cibles = [
    "Baroque", "Contemporary_Realism", "Cubism", 
    "Early_Renaissance", "Impressionism", "Ukiyo_e"
]

TARGET_SIZE = (256, 256)
NB_PAR_STYLE = 1000

os.makedirs(dest_folder, exist_ok=True)

print(f"🚀 Extraction de {len(styles_cibles) * NB_PAR_STYLE} images vers un dossier unique...")

total_copie = 0

for style in styles_cibles:
    chemin_src = os.path.join(source_root, style)
    
    if not os.path.exists(chemin_src):
        print(f"❌ Dossier source introuvable : {style}")
        continue

    # 1. Lister les fichiers du style actuel
    fichiers = [f for f in os.listdir(chemin_src) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # 2. Sélectionner 1000 images au hasard
    selection = random.sample(fichiers, min(NB_PAR_STYLE, len(fichiers)))
    
    print(f"🎨 Traitement du style {style}...")

    for nom_fichier in selection:
        try:
            with Image.open(os.path.join(chemin_src, nom_fichier)) as img:
                img = img.convert('RGB')
                img_resized = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
                
                # Pour éviter les doublons de noms de fichiers entre dossiers (ex: "1.jpg")
                # On renomme le fichier : "Baroque_1.jpg", "Cubism_1.jpg", etc.
                nouveau_nom = f"{style}_{os.path.splitext(nom_fichier)[0]}.jpg"
                
                img_resized.save(os.path.join(dest_folder, nouveau_nom), "JPEG", quality=85)
                total_copie += 1
        except Exception:
            continue

print(f"\n✨ TERMINÉ ! {total_copie} images de style sont réunies dans : {dest_folder}")