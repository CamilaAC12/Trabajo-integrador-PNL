
1. **Problema y destinatario.** Puede resolver preguntas sobre la intensificación de Julio-Agosto, Intensificación y EDTE. Para personar educativo.
2. **Corpus.** : 3 documentos n1°5 n°6 y n°10 sobre el régimen académico del portar ABC de de la dirección general de cultura y educación (DGCE) 
3. **Arquitectura.** El sistema implementa un flujo **RAG (Generación Aumentada por Recuperación)**
4. **Instalación local.** 
powershell
uv venv --python 3.12
.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
ollama pull granite4:latest
ollama pull bge-m3

Para ejecutar la app
.venv\Scripts\python.exe app.py

5. **Reindexación.** Al indexar los documentos, Chroma persiste la base vectorial en la carpeta local `chroma_db_dgcye_ra/` (colección `regimen_academico_pba_bge_m3`).
6. **Pruebas.** 

| Documento propio | Pregunta respondible | Página o sección esperada | ¿Qué recuperó? | ¿Qué interpretó el modelo? | Límite observado |
|---|---|---|---|---|---|
| Documento de apoyo n°5- La intensificación de las materias pendientes de aprobación y acreditación en base a la estructura curricular.pdf | ¿Cuántos modelos hay?| Página 3 (cuadro resumen con los 4 modelos) y Página 4 (mención explícita a "modelos 1, 2, 3 o 4"). | No recuperó el cuadro de la página 3; recuperó el fragmento de la página 4 donde se menciona la asignación de los modelos por parte del equipo directivo. | Respondió enumerando solo hasta el modelo 3, omitiendo el modelo 4 a pesar de que figuraba al final de la oración recuperada. | Extracción + Generación: PyPDFLoader suele desestructurar los cuadros/tablas (p. 3), haciendo que pierdan relevancia semántica; además, el modelo cortó la enumeración antes del cierre de la frase en la p. 4. |
| DOCUMENTO DE APOYO N°6 Intensificación_Profundización.pdf | ¿Cuántos momentos del año tienen el primer y el segundo grupo para aprender los saberes enseñados y las materias no aprobadas? | Páginas 11–13 (Organización de los tiempos y momentos de intensificación y profundización). | Recuperó las páginas 12, 13 y 11 del Documento de Apoyo N° 6. | Identificó los grupos y cifras (5 y 6 momentos), pero invirtió los valores (asignó 5 al primer grupo y 6 al segundo, cuando en el texto original es 6 y 5). | Corte + Generación: Al dividir el texto en chunks de 1100 caracteres, ambos grupos quedaron repartidos entre fragmentos consecutivos ([1] y [2]) y el LLM local cruzó los referentes. |
|DPESEC-Documento de Apoyo Nro 10_RA.pdf |- ¿Qué significa CCA? - ¿Qué se debe consignar en el RITE del primer cuatrimestre? | Página 2 (elaboración y actualización del RITE - Primer Cuatrimestre). | Con la sigla aislada ("¿Qué significa CCA?") no recuperó el fragmento pertinente. Al reformular con contexto ("¿Qué se debe consignar en el RITE..."), recuperó exactamente la página 2. | 1) Ante "¿Qué significa CCA?", respondió "No encuentro esa información en los documentos.".2) Ante la consulta sobre el RITE, recuperó el texto donde figura "Continúa, Con Avances (CCA)" junto con AA y CSA. | Recuperación por embeddings densos: Las siglas cortas y sin contexto (CCA) tienen baja densidad semántica para bge-m3; el modelo necesita palabras clave del entorno documental (RITE, primer cuatrimestre) para ubicar el fragmento. |

7. **Decisiones y límites.** 

* **Fragmentación (`chunk_size=1100`, `chunk_overlap=180`):** Los cortes se miden en caracteres. Usé fragmentos de `1100` con `180` de superposición porque con valores más chicos (`500/80`) se cortaban las oraciones y se perdían las condiciones de las normas. Igual, cuando dos datos parecidos quedan divididos entre dos fragmentos seguidos (como los momentos del primer y segundo grupo en el Doc. N° 6), el modelo a veces cruza los números.
* **Cantidad de fragmentos (`k=3`):** Con `3` fragmentos alcanza para darle contexto suficiente al modelo sin marearlo ni demorar la respuesta local. El límite es que si una página es muy larga (como la p. 1 del Doc. N° 10), dos de esos tres lugares pueden quedar ocupados por pedazos de la misma página.
* **Modelo de embeddings (`bge-m3`):** Funciona bien en español y corre local en Ollama. Lo que noté es que cuando busco una sigla sola (como *"¿Qué significa CCA?"*) no encuentra el fragmento, pero si pregunto con más contexto (*"¿Qué se debe consignar en el RITE..."*), lo recupera sin problema.
* **Fuentes y tipo de PDF:** Los tres documentos de la DGCyE son PDFs digitales con texto seleccionable.
* **Alcance de las citas:** El sistema solo responde con lo que está en esos recortes y no reemplaza la lectura completa de la normativa.

8. **Uso de IA.** Si la usaste, consigná para qué tareas, en qué momentos, qué modelos y qué comprobaste personalmente. Si no la usaste, podés indicarlo.

Uso de gémini para adaptar ciertas partes del código aplicados a los documentos seleccionados en específico.
Para interpretar resultados.
Para crear la app a partir de la que teniamos en clase.
Para completar el cuadro | Decisión | Explicación técnica en el proyecto | Prueba realizada y resultado observado |

Comprobación personal: Seleccioné los documentos, instalé y corrí todo el entorno local en Ollama, ejecuté cada pregunta en Gradio y verifiqué manualmente las respuestas y páginas citadas contra los PDFs originales.