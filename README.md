# Sethos RealVisXL + IP-Adapter — RunPod Serverless

Second moteur de **Administration > Outils créatifs > Créateur photo**.

- Base photoréaliste : `SG161222/RealVisXL_V5.0`
- Guidage : `h94/IP-Adapter`, variantes SDXL Plus Face et Plus
- Contrat : `sethos.realvisxl.ip-adapter.v1`
- Première référence : identité faciale, avec recadrage automatique du visage
- Seconde référence facultative : corps, pose, style ou décor
- Sortie : WebP encodé en base64 puis copié dans le stockage privé Sethos
- Sécurité : validation 18+, refus des termes liés aux mineurs et confirmation du consentement en amont

Ce moteur reconstruit une nouvelle photographie guidée par les références. Il est volontairement complémentaire de Qwen Image Edit, qui reste plus adapté aux retouches locales conservant exactement la composition d’origine.

## Endpoint RunPod

Le dépôt IP-Adapter minimal est intégré à l’image Docker. Dans le champ **Model** de l’endpoint RunPod, utiliser :

`SG161222/RealVisXL_V5.0`

RunPod monte le snapshot RealVisXL sous `/runpod-volume/huggingface-cache/hub/`. Le worker refuse tout téléchargement de poids pendant une tâche facturée.

Configuration recommandée : GPU de 24 Gio ou plus, `workersMin=0`, `workersMax=1`, disque conteneur de 50 Gio, arrêt après 5 secondes d’inactivité.

Les licences applicables restent celles de RealVisXL/OpenRAIL++ et d’IP-Adapter/Apache-2.0. L’utilisation doit rester limitée à des personnes majeures, consentantes, et à des images pour lesquelles l’utilisateur possède les droits nécessaires.
