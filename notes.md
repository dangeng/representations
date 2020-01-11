- success
    - conv encoder -> latent 4x32x32 -> conv decoder
    - conv encoder -> fc 4096x4096 -> conv decoder

- some vague signs of life
    - conv encoder -> fc 4096->1024 || (wait i tried again and it didn't work???)
    
- fail
    - conv encoder -> fc latent 2 -> conv decoder
    - conv encoder -> fc latent 4 -> conv decoder
    - conv encoder -> fc latent 128 -> conv decoder
    - conv encoder -> fc 4096->1024->256 ||
    - conv encoder -> fc 4096->2048||
