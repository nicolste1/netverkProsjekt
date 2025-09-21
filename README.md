# Nevralt Nettverk for MNIST data


# Bakgrunn
Prosjekt oppgave i TDT4102 objektorientert programering på NTNU. Prosjektet er laget i C++ i den hensikt av å utvikle kunnskap om nevrale nettverk. Det nevrale nettverket er bygd fra bunnen og bygger på kode lært i faget, kode og kunnskap fra Michael Nielsens bok "Introduces neural networks and deep learning": https://goo.gl/Zmczdy, samt videoer fra 3blue1brow om deep Learning. 

# Intro
Første del av prosjektet ble laget med 2D matriser(slowNetwork) noe som er grunnlaget for nettverket, videre er netverket blitt optimlaisert med 1D matriser (fastNetwork).
Hovedfunksjon til netverket er å klassifisere håndskrevne sifre fra MNIST.

# Innhold
- `sourcecode/fastNetwork/`: Optimalisert nettverk med tilhørende matriseimplementasjon.
- `sourcecode/slowNetwork/`: Tregt netverk med tilhørende matriseimplementasjon. 
- `sourcecode/MNIST/`: Kode for å laste inn, og manipulere dat, samt vise og tegne MNIST-data.
- `sourcecode/xor/`: XOR-eksempel for test av netverk.
- `sourcecode/dataPath`: Oversikt over PATHs og tilhørende funskoner for bruk av lagret data. 
- `data/`: Ferdiglagrede datasett fra MNIST samt fildata for netverk. 
- `main.cpp`: Startpunktet for programmet.

# Funksjoner
- Trene nettverket fra gitt data.
- Laste og lagre netverk. 
- Hente ut trenings data fra MNIST som netverk trener på.
- Sjekke Netverk med test data fra MNIST kan ses visuelt i GUI. 
- Tegn egne sifre og få dem klassifisert med nevralt nettverk.
- Trene på egen håndskrift, som blir augmentert og oversamplet deretter trent på, noe som skjer i en separat thread.

# Mulige forbedringer
- Ved trening på egen kan og bør det legges til en funksjon som (Bruker sandpapir) på tegningene for å lage flere nesten like tegninger.
- MNIST funksjoner samt visuel MNIST burde lagges inn i klasser for å utnytte fordeler med OOP.
- Funksjonen for å lage MNIST netverk kan bruke flere og støre lag for å få en mer nøyaktig modell. 

# Kompilering og kjøring

Prosjektet bruker **Meson**. For å kompilere:

```bash
meson setup builddir --buildtype debug
meson compile -C builddir
./builddir/program.exe

