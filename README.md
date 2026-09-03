# Sethos RealVisXL + IP-Adapter — RunPod Serverless

Second moteur de **Administration > Outils créatifs > Créateur photo**.

- Base photoréaliste : `SG161222/RealVisXL_V5.0`
- Guidage : `h94/IP-Adapter`, variantes SDXL Plus Face et Plus
- Contrat : `sethos.realvisxl.ip-adapter.v1`
- Première référence : identité faciale, avec recadrage automatique du visage
- Seconde référence facultative : corps, pose, style ou décor, avec rôle explicite `general` ou `body_pose_clothed`
- Réglage `prompt_adherence` : `strict` (texte prioritaire), `balanced` ou `reference`
- Sortie : WebP encodé en base64 puis copié dans le stockage privé Sethos
- Sécurité : validation 18+, refus des termes liés aux mineurs et confirmation du consentement en amont
- Canary LoRA privé facultatif : artefact chargé depuis le volume persistant, identifiant borné et empreinte SHA-256 vérifiée avant toute génération

Ce moteur reconstruit une nouvelle photographie guidée par les références. Il est volontairement complémentaire de Qwen Image Edit, qui reste plus adapté aux retouches locales conservant exactement la composition d’origine.

Le mode `strict`, sélectionné par défaut, envoie l’instruction utilisateur aux deux encodeurs SDXL, renforce le guidage textuel et limite l’influence de la référence hors visage. Le niveau de préservation `identity` utilise parallèlement un recadrage facial serré afin de réduire la dérive des traits.

Pour une demande de nudité, la photo source complète n’est plus réinjectée dans le second IP-Adapter : seul son recadrage facial sert à l’identité. Une référence `body_pose_clothed` reste un guide structurel faible afin d’éviter de recopier sa tenue ; elle doit correspondre précisément à la pose demandée.

## Endpoint RunPod

Le dépôt IP-Adapter minimal est intégré à l’image Docker. Dans le champ **Model** de l’endpoint RunPod, utiliser :

`SG161222/RealVisXL_V5.0`

RunPod monte le snapshot RealVisXL sous `/runpod-volume/huggingface-cache/hub/`. Le worker refuse tout téléchargement de poids pendant une tâche facturée.

Le dépôt RealVisXL publie simultanément un pipeline Diffusers et un checkpoint
FP16 unique. Le worker privilégie le pipeline complet. Si RunPod expose un
cache Diffusers contenant seulement les poids, il superpose la configuration
légère intégrée à l’image Docker et lie les poids sans les recopier. Si le cache
contient plutôt le checkpoint `RealVisXL_V5.0_fp16.safetensors`, il utilise le
chargeur fichier unique. Ces reprises ne téléchargent rien pendant le job.

Un LoRA de personnage validé se place dans
`/runpod-volume/sethos-lora/<artifact_id>/pytorch_lora_weights.safetensors`.
Le site transmet `artifact_id`, le token du personnage, la force du LoRA et le
SHA-256 attendu. Le worker refuse un chemin hors volume, un fichier absent ou
une empreinte différente. Sans ces quatre champs, le chemin historique reste
strictement inchangé.

Configuration recommandée : GPU de 24 Gio ou plus, `workersMin=0`, `workersMax=1`, disque conteneur de 50 Gio, arrêt après 5 secondes d’inactivité.

Les licences applicables restent celles de RealVisXL/OpenRAIL++ et d’IP-Adapter/Apache-2.0. L’utilisation doit rester limitée à des personnes majeures, consentantes, et à des images pour lesquelles l’utilisateur possède les droits nécessaires.
