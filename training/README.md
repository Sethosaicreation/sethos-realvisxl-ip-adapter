# Entraînement du LoRA de personnage

Le Studio exporte un ZIP seulement après sélection humaine d'au moins 16 images
uniques et équilibrées. Le ZIP contient `images/metadata.jsonl`, compris par le
chargeur `imagefolder` de Hugging Face Datasets, et `manifest.json` avec le token
de déclenchement.

1. Extraire le ZIP dans un dossier privé.
2. Utiliser un environnement GPU contenant le dépôt Diffusers à la révision
   `d6bfaa71b858f32bdc54ab9868e0385c093f1122`, `accelerate`, `datasets` et
   `peft`.
3. Lancer `training/train_character_lora.sh` avec `DATASET_DIR`, `OUTPUT_DIR`
   et `INSTANCE_TOKEN`.
4. Inspecter les images de validation. Rejeter le modèle si le visage dérive,
   si une pose ou un décor est mémorisé, ou si les proportions deviennent moins
   stables.
5. Copier uniquement `pytorch_lora_weights.safetensors` dans
   `/runpod-volume/sethos-lora/<artifact_id>/` puis renseigner dans le Studio
   l'identifiant, le SHA-256 affiché par le script et une force initiale de
   `0.72` à `0.82`.

Le nombre de pas par défaut (1 600) est volontairement prudent. Augmenter le
nombre de pas n'est pas une correction universelle : un surapprentissage copie
les poses, les chambres et les défauts des images d'entraînement. Le canary ne
doit être activé qu'après comparaison à seeds et prompts identiques contre le
workflow historique.
