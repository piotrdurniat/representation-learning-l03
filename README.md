# Assignment l03-24

Repozytorium zawiera notebooki zawierające zadanie na ten tydzień. 
Wykonaj polecenia w każdym notebooku (rozszerzenie .ipynb). 
Następnie zacommituj pracę i wypushuj na GitHuba. 
Prowadzący sprawdzi wykonane zadanie po jego terminie.

Wyniki zostaną opublikowane w plikach feedbacku `*.html`. 
Konieczne będzie pobranie zmian i otwarcie pliku w przeglądarce (GitHub nie renderuje plików HTML).

----
## Konfiguracja środowiska

W laboratorium wykorzystywane są zeszyty [Jupyter Notebook](https://jupyter.org/).
Środowisko można skonfigurowac "własnoręcznie" lub skorzystać z obrazów docker. Kod przygotowany został zarówno pod
CPU jak i GPU (kod działa na dowolnym urządzeniu, jednak trzeba wcześniej zainstalować zależności odpowiednie do urządzenia).
W celu ulatwienia instalacji można skorzystać z pliku `Makefile` odpowiednio ustawiając
sobie `DEVICE=cpu|gpu`

### Docker

1. Budowanie obrazu

```shell
make build_image DEVICE=<cpu|gpu>
```

2. Uruchamianie kontenera (domyślnie na porcie `8888`, można zmodyfikować ustawiając w komendzie zmienną `PORT`, np. `PORT=8080`)

```shell
  make run_container DEVICE=<cpu|gpu>
```

### Lokalne środowisko

Należy użyć języka **Python w wersji 3.11**, zaleca się używania [wirtualnych środowisk](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) `virtualenv` lub `conda`.
W przypadku wystąpienia problemów z instalacją niektórych paczek spróbuj najpierw wykonać upgrade `pip install -U pip` 

```shell
   make install DEVICE=<cpu|gpu>
```

Po zainstalowaniu paczek w środowisku, należy uruchomić serwer Jupyter poleceniem:

```shell
   jupyter <notebook|lab>
```

Po uruchomieniu w przeglądarce otworzy się strona Jupyter, z której można wybrać plik zeszytu (o
rozszerzeniu `*.ipynb`). Jeżeli strona nie otworzyła się automatycznie, można wykorzystać link, który pojawi się w konsoli.
