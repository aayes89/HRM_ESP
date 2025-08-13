# HRM_ESP
Inspirado en el estilo de un "HRM" (High-level Reasoning Model), diseñado específicamente para texto en español.<br>
Define un modelo de lenguaje basado en Transformer con un enfoque de dos niveles: un Planner (planificador) de alto nivel que genera un embedding de "plan" a partir de un prompt, y un Executor (ejecutor) de bajo nivel que genera texto condicionado por ese plan.<br>
Está diseñado para:

* <b>Preentrenamiento</b>: Entrenar el modelo como un modelo de lenguaje puro en un corpus de texto en español.
* <b>Fine-tuning</b>: Ajustar el modelo con prompts y respuestas específicas (formato TSV: prompt \t target) o como modelo de lenguaje si no hay respuestas.
* <b>Generación</b>: Generar texto a partir de un prompt utilizando técnicas de muestreo como top-k, top-p y temperatura.

El modelo utiliza un tokenizador byte-level (UTF-8) para manejar texto en español, lo que permite una representación robusta de caracteres y es especialmente útil para idiomas con caracteres especiales.

# Componentes principales
<h3>Tokenizador (ByteTokenizer)</h3>

Función: Convierte texto en español a una secuencia de enteros (tokens) y viceversa.
<p>Detalles:
  
* <b>Usa codificación UTF-8 a nivel de bytes (0 a 255) más 4 tokens especiales</b>: PAD (256), BOS (257, inicio de secuencia), EOS (258, fin de secuencia) y PLAN (259, para inyectar el embedding del plan).
* <b>encode</b>: Convierte un string a una lista de enteros (bytes UTF-8, con BOS y EOS opcionales).
* <b>decode</b>: Convierte una lista de enteros a un string, ignorando tokens especiales.
* <b>Ventaja</b>: Es simple, no depende de un vocabulario predefinido y maneja bien caracteres no estándar.</p>

<h1>Carga de datos (load_corpus_lines, dataset_lm, dataset_tsv)</h1>

Función: Carga y preprocesa datos para entrenamiento.
<p>Detalles:<br>
  
* load_corpus_lines: Lee un archivo de texto línea por línea (UTF-8, ignorando errores).
* dataset_lm: Convierte un archivo de texto en una lista de secuencias tokenizadas para preentrenamiento (modelo de lenguaje puro).
* dataset_tsv: Lee un archivo TSV (prompt \t target) para fine-tuning. Si no hay target, usa el propio prompt como objetivo (modo LM puro).
Los datos se tokenizan con ByteTokenizer y se preparan para el modelo.</p>

<h1>Componentes del Transformer</h1>
<p>
  El modelo utiliza bloques estándar de Transformer, adaptados para su arquitectura de dos niveles:

* <b>RMSNorm</b>: Normalización de capa basada en la raíz cuadrada media (RMS), alternativa a LayerNorm, más eficiente.
* <b>MHA (Multi-Head Attention)</b>: Atención multi-cabeza, con soporte para atención causal (para el decodificador) y atención cruzada (para incorporar el plan, aunque no se usa en este caso).
* <b>FFN (Feed-Forward Network)</b>: Red feed-forward con activación GEGLU (Gated Linear Unit), que combina una activación GELU con una puerta lineal.
* <b>DecoderBlock</b>: Bloque Transformer causal con atención propia y FFN.
* <b>EncoderBlock</b>: Bloque Transformer no causal (para el Planner).
</p>

<h3>Modelo HRM (HRMModel)</h3>
El modelo se divide en dos componentes principales:
<h3>Planner</h3>

* Entrada: Prompt tokenizado (prompt_ids).
* Función: Genera un embedding de "plan" que resume el prompt.

<b>Estructura</b>
<b>Embedding de tokens y posiciones.</b>
* n_layers_enc bloques Transformer no causales (EncoderBlock).
* Mean pooling sobre la secuencia para obtener un vector resumen.
* Proyección a plan_dim (dimensión del plan).
Opcionalmente, usa Gumbel-Softmax para discretizar el plan en un codebook de n_plan_codes vectores.

Salida: Un vector de plan (plan_vec) y, si se usa Gumbel-Softmax, logits para el codebook.

<h3>Executor:</h3>

* Entrada: Secuencia de entrada (y_in) y el vector de plan (plan_vec).
* Función: Genera texto de manera autoregresiva, condicionado por el plan.

<b>Estructura:</b>
<b>Embedding de tokens y posiciones.</b><br>

* Proyección del plan_vec a un "token" sintético ([PLAN]) que se concatena como prefijo.
* n_layers_dec bloques Transformer causales (DecoderBlock).
* Capa final (lm_head) para predecir el siguiente token.
 
Salida: Logits para la distribución de probabilidad sobre el vocabulario.

<h3>Entrenamiento</h3>
<p>
<b>Preentrenamiento (train_lm)</b>:
  
* Usa un corpus de texto en español (archivo TXT).
* Entrena el modelo como un modelo de lenguaje puro, prediciendo el siguiente token en secuencias de longitud max_len.
* El Planner ve la misma secuencia que el Executor (condición débil).
* Calcula la pérdida de cross-entropy, con una pequeña penalización opcional para el codebook (si se usa Gumbel-Softmax).
* Guarda checkpoints por época y un checkpoint final.

<b>Fine-tuning (train_ft)</b>:

* Carga un modelo preentrenado desde un checkpoint.
* Usa un archivo TSV (prompt \t target) para ajustar el modelo a tareas específicas.
* Si no hay target, entrena como LM puro.
* Similar al preentrenamiento, pero con prompts y targets explícitos.

<b>Optimización</b>:

* Usa AdamW con learning rate configurable.
* Soporta mixed precision (AMP) en CUDA para mayor eficiencia.
* Aplica dropout para regularización.
</p>

<h3>Generación (generate)</h3>
Función: Genera texto a partir de un prompt.
<p>Detalles:

Tokeniza el prompt y genera el plan_vec con el Planner.<br>
Usa el Executor para generar tokens autoregresivamente, comenzando con BOS.

Aplica muestreo con:

* <b>Top-k</b>: Selecciona los k tokens más probables.
* <b>Top-p (nucleus sampling)</b>: Selecciona un conjunto de tokens cuya probabilidad acumulada supera p.

<b>Temperatura</b>:
* Controla la aleatoriedad (valores bajos hacen el muestreo más determinista).
* Detiene la generación al alcanzar max_new_tokens o el token EOS.
* Decodifica la salida a texto usando el tokenizador.<br>
</p>

<h3>Interfaz de línea de comandos (cli)</h3>
<b>Comandos</b>:

* <b>pretrain</b>: Entrena el modelo en un corpus de texto.
* <b>finetune</b>: Ajusta un modelo preentrenado con un archivo TSV.
* <b>generate</b>: Genera texto a partir de un prompt y un checkpoint.
 
<b>Argumentos</b>:
* Configuraciones como batch_size, max_len, lr, d_model, n_heads, epochs, etc.
* Opciones para desactivar CUDA (--cpu) o Gumbel-Softmax (--no_gumbel).
* Parámetros de generación como temperature, top_k, top_p.
