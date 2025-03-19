# upo-AIContrastive
Progetto per la tesi magistrale.\
Utilizzo di Loss Contrastive per migliorare le predizioni di un Classificatore.

> [!IMPORTANT]
> La versione di Python usata è la [3.10.12](https://www.python.org/downloads/release/python-31012/).\
> Per installare usare il comando: `pip install -r requirements.txt`

### Dataset
  - [ ] Splittare il dataset in due parti: L e U.
  - [ ] Usare tutto il Dataset per l'auto encoder
  - [ ] Usare L per addestrare il classificatore
  - [ ] Prendere i risultati e fare le coppie Positive (sbagliati + classe giusta) e Negative (sbagliati + classe sbagliata). Es. A{a,a,a,b} e B{b,b,a} → (Ab,Bb)+ (Ab,Aa)-
### Autoencoder per fare embedding
  - [ ] Cercare Online un modello autoencoder per radiografie (se esiste usarlo). Trovato solo auto encoders per denoising
  - [ ] Altrimenti fare una architettura usando 3/4 Conv + dense
  - [ ] L’embedding può essere di 256/128 valori. (metterlo come parametri iniziale)
  - [ ] Usare l’intero dataset per l’addestramento
  - [ ] Salvare la rete
  - [ ] Salvare gli embedding e i label
### Classificatore 
  - [ ] Fare un semplice classificatore che prende in input gli embedding
  - [ ] Il classificatore può essere un semplice MLP con in fondo un Batch Norm
  - [ ] Nel caso usare SVM
  - [ ] Usare i dati provenienti da L per l’addestramento
  - [ ] Salvare la rete
  - [ ] Salvare gli embedding e i risultati del classificatore
### Siamese NET
  - [ ] Creare le coppie Positive e Negative.
    - [ ] Positive: un caso sbagliato (A → B) e un caso corretto (A → A)
    - [ ] Negative: un caso sbagliato (A → B) e un caso corretto (B → B)
    - [ ] (?) Prendere solo un subset di quelli etichettati sbagliati
    - [ ] (?) Prendere solo un subset delle coppie generate
  - [ ] La rete è una semplice MLP di uno strato più piccolo (comprime) e un Batch Norm
  - [ ] La rete prende in input un solo embedding, ma viene “duplicata”
  - [ ] Usare le coppie di embedding come input
  - [ ] Il risultato è un embedding corretto
  - [ ] Usare la Loss Contrastive (quella a batch positivi/totali)
  - [ ] Mettere la temperatura ma inizialmente lasciarla a 1
### Test
- [ ] Usare il dataset Unlabeled
- [ ] Fare la predizione con embedding semplici
- [ ] Fare predizione con embedding “corretti”
