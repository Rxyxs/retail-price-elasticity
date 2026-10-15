[English](README.md) | **Español**

# ¿Cuánto mueve la demanda de supermercado un cambio de precio?

![CI](https://github.com/Rxyxs/retail-price-elasticity/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Dos años de compras reales de supermercado de 2.500 hogares (dunnhumby, *The Complete Journey*),
con el precio que pagó cada hogar, el descuento de la tarjeta de fidelidad, y si cada producto
estaba en exhibición en tienda o en el folleto semanal. Estimo la elasticidad precio de la demanda
de 580 productos envasados en 12 categorías y la convierto en la pregunta que se hace un jefe de
categoría: **¿cuándo se paga sola una rebaja?**

Lo difícil es que los precios no se mueven al azar. Los productos entran en oferta las mismas
semanas en que se exhiben y salen en el folleto, y los compradores se abastecen antes de la
siguiente oferta. Cada paso de abajo quita uno de esos sesgos.

## Resultados

| Estimador | Elasticidad | |
|---|---|---|
| OLS agrupado | -0,21 | compara productos caros y baratos, no cambios de precio |
| + efectos fijos de producto y semana | -2,10 | cada producto contra su propio precio habitual |
| + exhibición y folleto | -1,47 | las promociones suben las ventas por sí solas |
| + precio de la semana anterior | -1,50 | el abastecimiento mueve ventas entre semanas |
| **Double ML (LightGBM con cross-fitting)** | **-1,50** (EE 0,06) | controles flexibles, misma respuesta |

![Escalera de estimadores](docs/img/ladder.png)

**Ignorar las promociones exagera la sensibilidad al precio en un 40%.** Solo con efectos fijos,
una baja de precio de 1% parece sumar 2,1% a las ventas; al controlar exhibición y folleto suma
1,5%. El resto era la promoción, atribuida al precio.

**Las rebajas le piden prestado a la semana siguiente.** Un producto que estuvo barato la semana
pasada vende menos esta semana (coeficiente del precio relativo rezagado +0,11, EE 0,03): parte de
lo que "vende" una oferta son compras adelantadas.

**Double ML coincide con el modelo lineal con controles.** Residualizar precio y ventas sobre
promociones, rezagos, estacionalidad y categoría con LightGBM (folds agrupados por producto) da el
mismo -1,50, así que en este panel los controles lineales ya capturan la confusión. Queda el 74% de
la variación de precio dentro de cada producto después de quitar lo que predicen los controles.

## Por categoría

![Elasticidad por categoría](docs/img/by_category.png)

| Categoría (productos) | Elasticidad [IC 95%] | Ingresos si el precio sube 5% | Una rebaja de 10% se paga sobre un margen bruto de |
|---|---|---|---|
| Cerveza (9) | -2,11 [-2,61, -1,62] | -5,3% | 50% |
| Pizza congelada (43) | -2,07 [-2,34, -1,80] | -5,1% | 51% |
| Galletas y conos (9) | -2,01 [-2,19, -1,83] | -4,8% | 52% |
| Bebidas (111) | -1,94 [-2,16, -1,73] | -4,5% | 54% |
| Galletas saladas (15) | -1,84 [-2,34, -1,35] | -4,0% | 57% |
| Leche (25) | -1,79 [-2,22, -1,36] | -3,8% | 58% |
| Cereal (39) | -1,77 [-1,99, -1,55] | -3,7% | 59% |
| Snacks en bolsa (58) | -1,48 [-1,74, -1,22] | -2,3% | 69% |
| Queso (65) | -1,32 [-1,53, -1,11] | -1,5% | 77% |
| Pan (51) | -1,16 [-1,40, -0,91] | -0,8% | 87% |
| Sopa (60) | -1,05 [-1,31, -0,78] | -0,2% | 96% |
| Yogur (95) | -0,19 [-0,55, 0,17] | +4,0% | nunca |

**Una rebaja casi nunca se paga sola con su propio producto.** Bajar el precio un 10% sube la
utilidad solo si el volumen extra compensa el margen perdido por unidad. Incluso en la categoría
más sensible, cerveza, eso exige un margen bruto sobre 50%; en pan y sopa, 87-96%. Las rebajas de
supermercado que tienen sentido se pagan por otro lado: tráfico a la tienda, tamaño de la canasta
o financiamiento del proveedor, cosas que este análisis por producto no mide.

**Las categorías de impulso y antojo son las más elásticas** (cerveza, pizza congelada, galletas,
bebidas); **los básicos están cerca de la elasticidad unitaria** (pan, sopa), y en esos dos el
intervalo incluye -1, así que los datos no dicen si subir el precio ganaría o perdería ingresos. El
yogur es la excepción: sus ventas casi no responden al precio.

## Método

```
transaction_data (2,6 M líneas) + product + causal_data (36 M filas de promoción tienda-semana)
   │
   ├── categorías envasadas que se venden por unidad; productos comprados en 60+ de 102 semanas
   ├── panel producto-semana: unidades, precio pagado tras el descuento de fidelidad, precio de
   │   lista, número de tiendas con exhibición / folleto, precio relativo de la semana anterior
   │
   ├── escalera OLS con efectos fijos de producto y semana, errores agrupados por producto
   ├── double ML: residuos dentro de producto sobre promociones, rezagos, estacionalidad,
   │   categoría (LightGBM, 5 folds agrupados por producto), EE robustos por clúster
   └── por categoría: elasticidad, respuesta de ingresos, margen de equilibrio para una rebaja de 10%
```

`causal_data` solo lista tienda-semanas con alguna promoción, así que una proporción de sus filas
sería una proporción entre tiendas ya promocionadas; el panel cuenta tiendas con promoción.

## Datos

dunnhumby, *The Complete Journey*, vía Kaggle
[frtgnn/dunnhumby-the-complete-journey](https://www.kaggle.com/datasets/frtgnn/dunnhumby-the-complete-journey),
licencia DbCL 1.0. Descargados en octubre de 2026; no se redistribuyen aquí.

## Reproducir

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
python -m elastic.panel     # descarga con ~/.kaggle/kaggle.json y arma data/panel.parquet
python -m elastic.model     # ~5 min
python -m elastic.figures
pytest -q tests
```

## Limitaciones

- Son elasticidades de corto plazo identificadas sobre todo por ofertas temporales. Un cambio de
  precio permanente suele mover menos la demanda que una oferta.
- Se descartan las semanas sin compras en el panel, porque no hay precio observado; en productos de
  baja rotación esto se inclina hacia semanas en que el producto estaba barato.
- No se modelan efectos cruzados (una oferta de una bebida que le quita ventas a otra), así que los
  márgenes de equilibrio ignoran la canibalización además del tráfico.
- 2.500 hogares son un panel, no las ventas completas de las tiendas.

## Licencia

Código: MIT. Datos: DbCL 1.0 (dunnhumby vía Kaggle), no se redistribuyen.
