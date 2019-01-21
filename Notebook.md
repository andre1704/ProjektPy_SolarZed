
# Projekt SolarZed - zaawansowana eksploracja danych 
### autor: Andrzej Słowiński
### data: 21.01.2019


## Wstęp
Celem proejktu była predykcja ilości energii wyprodukowanej przez panele słoneczne, na podstawie danych treningowych oraz testowych. Z uwagi na typ danych który był na wejściu oraz wyjściu jest to zadanie gdzie należało rozważać regresję. Trenując algorytm oraz szukając najlepszego rozwiązania sprawdzany jest rmse dla regresji liniowej, oraz dla różnych wartości estymatora w RandomForestRegressor.

### Biblioteki oraz wczytanie danych
Dane zostały wczytane przy pomocy biblioteki pandas, z pakietu sklearn dla modelu użyte zostały biblioteki LinearRegression, RandomForestRegressor, obliczanie rmse użyta została biblioteka mean_squared_error.


```python
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


train = pd.read_csv('Data/train.csv',sep=',')
test = pd.read_csv('Data/test.csv', sep=',')

```

### Sprawdzenie danych
Dane treningowe oraz testowe zostały przeze mnie sprawdzone czy nie zawierają wartyości pustych, z uwagi że nie było takich wartości nie wymagana była obróbka danych pod kątem wartości pustych.


```python
train.isnull().sum()
test.isnull().sum()
```




    id                      0
    idsito                  0
    idmodel                 0
    idbrand                 0
    lat                     0
    lon                     0
    ageinmonths             0
    anno                    0
    day                     0
    ora                     0
    data                    0
    temperatura_ambiente    0
    irradiamento            0
    pressure                0
    windspeed               0
    humidity                0
    icon                    0
    dewpoint                0
    windbearing             0
    cloudcover              0
    tempi                   0
    irri                    0
    pressurei               0
    windspeedi              0
    humidityi               0
    dewpointi               0
    windbearingi            0
    cloudcoveri             0
    dist                    0
    altitude                0
    azimuth                 0
    altitudei               0
    azimuthi                0
    pcnm1                   0
    pcnm2                   0
    pcnm3                   0
    pcnm4                   0
    pcnm5                   0
    pcnm6                   0
    pcnm7                   0
    pcnm8                   0
    pcnm9                   0
    pcnm10                  0
    pcnm11                  0
    pcnm12                  0
    pcnm13                  0
    pcnm14                  0
    pcnm15                  0
    irr_pvgis_mod           0
    irri_pvgis_mod          0
    dtype: int64



### Obróbka danych wejściowych
W danych wejściowych znajdowała się kolumna z datą której format uniemożliwaiłby predykcję, dlatego też tworząc funkcję splitDate dane zostały odpowidenio podzielone na kolejne lata, miesiące, dni, godziny.



```python
def splitDate(x, date_part):
    if(date_part == "year"):
        y = datetime.strptime(x, '%m/%d/%Y %H:%M').year
    elif(date_part == "month"):
        y = datetime.strptime(x, '%m/%d/%Y %H:%M').month
    elif(date_part == "day"):
        y = datetime.strptime(x, '%m/%d/%Y %H:%M').day
    elif(date_part == "hour"):
        y = datetime.strptime(x, '%m/%d/%Y %H:%M').hour
    return int(y)

train['year']= train['data'].apply(lambda x: splitDate(x, "year"))
train['month']= train['data'].apply(lambda x: splitDate(x, "month"))
train['day_']= train['data'].apply(lambda x: splitDate(x, "day"))
train['hour']= train['data'].apply(lambda x: splitDate(x, "hour"))

test['year']= test['data'].apply(lambda x: splitDate(x, "year"))
test['month']= test['data'].apply(lambda x: splitDate(x, "month"))
test['day_']= test['data'].apply(lambda x: splitDate(x, "day"))
test['hour']= test['data'].apply(lambda x: splitDate(x, "hour"))
```

W danych treningowych pojawiała się kolumna która była wartością energii paneli słonecznyhch co było wartością wyjściową, dlatego też kolumna ta została przypisana do nowej zmiennej a z danych treningowych usunięta tak jak i stara nie przetworzona kolumna daty.


```python
X_train = train
X_train = X_train.drop("kwh",1)
X_train = X_train.drop("data",1)
y_train = train.loc[:,'kwh']
```


```python
X_test = test.drop("data",1)
```

### Dane testowe i treningowe powtórny podział
Z uwagi na to, że musiałem stwierdzić czy dany algorytm predykcji będzie odpowiedni a jednym z sposobów na to jest rmse gdzie potrzebne są dane prawdziwe oraz dane predykowane z danego zbioru testowego podzieliłem zbiór traningowy na podziobry testowe oraz treningowe (X1_train, X1_test, y1_train, y1_test) dzięki czemu mogłem trenować model i badać czy wartość rmse jest dobra. 


```python
X1_train, X1_test, y1_train, y1_test = train_test_split(X_train, y_train, test_size=0.2, random_state=0)  
```

### Regresja, uczenie modelu
Projekt obejował wykorzystanie regresji, testowałem w tym zakresie regresję liniową, niestety napotkałem na problem ujemnych wartości które uzyskiwałem po zastosowaniu predykcji, wartość energii nie mogła być ujemna dlatego też przypisałem do wszystkich wartości wyjściowych 0 w przypadku wartości poniżej 0. Po tej operacji wyliczona została wartość rmse dla regresji liniowej.
Z napotkanym problemem wartości ujemnycfh poradził sobie model RandomForestRegressor który wymagał jednak jako wartości początkowych informacji o wartości estymatora, dlatego też zbadałem kolejno dla 10 , 50 , 100  czy błąd rmse będzie dla predykowanych danych się poprawiał.
Uzyskane dane rmse oraz coeficient zostały zapisane do tabeli array_coef_squared_error.


```python
def zeroAssign(x):
    if(x>=0):
        return x
    else:
        return 0

array_coef_squared_error = []
tab_estimators = [10 , 50, 100]

reg = LinearRegression(copy_X=True, fit_intercept=True, normalize=True).fit(X1_train, y1_train)
y_pred = reg.predict(X1_test)
pred_d = {'Id': X1_test['id'] , 'Predicted': y_pred}
predicted_data = pd.DataFrame(data=pred_d)
predicted_data['Predicted'] = predicted_data['Predicted'].apply(lambda x: zeroAssign(x))  
array_coef_squared_error.append(["linear regression",reg.score(X1_test, y1_test), mean_squared_error(y1_test,predicted_data['Predicted'])])


for i in tab_estimators:
    reg = RandomForestRegressor(max_depth=14,n_estimators=i).fit(X1_train, y1_train)
    
    y_pred = reg.predict(X1_test)
    name = "random forest"+str(i)
    coef = reg.score(X1_train, y1_train)
    rmse = mean_squared_error(y1_test,y_pred)
    array_coef_squared_error.append([name,coef,rmse])



```

### Tabela podsumowująca


```python
from astropy.table import Table, Column
summary = pd.DataFrame(array_coef_squared_error)
summary.columns=['nazwa algorytmu','coefficient ','rmse']
                     
print(summary)
```

         nazwa algorytmu  coefficient       rmse
    0  linear regression      0.810993  0.008046
    1    random forest10      0.963922  0.002872
    2    random forest50      0.966571  0.002705
    3   random forest100      0.966864  0.002707
    

### Predykcja danych
Dane testowe zostały wykorzystane po nauczeniu modelu do wygenerowania predykowanych danych oraz ich zapisania do pliku csv.


```python
y_pred = reg.predict(X_test)
pred_d = {'Id': X_test['id'] , 'Predicted': y_pred}
predicted_data = pd.DataFrame(data=pred_d)  
predicted_data['Predicted'] = predicted_data['Predicted'].apply(lambda x: zeroAssign(x))
predicted_data.to_csv('Data/submission.csv',sep=',' ,index = False)

```

## Podsumowanie
W przypadku RandomForestRegressor testowałem różne wartości n_estimators ostatecznie, wartości rmse nie poprawiały się wystarczającą aby dalej wartość tą zwiększać. W przypadku parametru max_depth dobrany on został ręcznie, po kilku próbach przy coraz większych jego wartościach czas tworzenia modelu wydłużał się dlatego wybrana została największa wartość którą udało się przetestować.
