🧠 Clasificación de Meses de edad en un Ostion con Redes Neuronales
📌 Introducción
Este proyecto utiliza un modelo de red neuronal profunda para predecir el mes basado en medidas de longitud, ancho y peso de una especie de ostion. Implementa técnicas de preprocesamiento, entrenamiento y evaluación con TensorFlow/Keras y Scikit-learn. con el fin de utilizarlo para la clasificacion de ostiones en CIDIR.
🛠 Instalación y Configuración
Asegúrate de tener instaladas las siguientes librerías:
pip install pandas numpy matplotlib seaborn tensorflow scikit-learn openpyxl

📂 Datos y Preprocesamiento
El dataset Dataset.xlsx contiene las siguientes características:
- Longitud
- Largo
- Ancho
- Peso
- Mes (variable objetivo)
📑 Pasos:
1️⃣ Conversión de etiquetas: Se utiliza LabelEncoder para transformar los meses en valores numéricos y to_categorical para realizar one-hot encoding.
2️⃣ Normalización: Se aplica StandardScaler para escalar los valores.
3️⃣ División del dataset: Se separa en conjuntos de entrenamiento y prueba (train_test_split).

🏗 Creación del Modelo
Se implementa una red neuronal profunda con las siguientes capas:
- Dense(64, activation='relu')
- Dropout(0.3)
- Dense(32, activation='relu')
- Dropout(0.3)
- Dense(6, activation='softmax') (salida categórica para 6 meses)
🔧 Compilación y Entrenamiento
El modelo se compila con:
optimizer = Adam(learning_rate=0.0005)  
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


Entrenamiento:
model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_split=0.2)


Se utiliza Dropout para reducir el overfitting y mejorar la capacidad de generalización.

🎯 Evaluación del Modelo
El modelo se evalúa con el conjunto de prueba:
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Pérdida en test: {loss}, Precisión: {accuracy}')


Se observa el desempeño en términos de precisión.

🔮 Predicciones
Para predecir el mes basado en una nueva entrada:
entrada = np.array([[8.5, 1.3, 3.3, 7.5]])  
entrada_scaled = scaler.transform(entrada)  
prediccion = model.predict(entrada_scaled)  
mes_index = np.argmax(prediccion)  
mes_predicho = label_encoder.inverse_transform([mes_index])[0]  
print(f'El mes predicho es: {mes_predicho}')



📈 Visualización de Datos
Se incluyen gráficos para analizar las características:
✅ Histogramas
✅ Boxplots
✅ Scatter plots
✅ Matriz de Correlación
Ejemplo de análisis con Seaborn:
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Matriz de Correlación')
plt.show()



💾 Guardado y Carga del Modelo
El modelo y los preprocesadores se guardan para su reutilización:
model.save('modelo_clasificacion_mes.keras')
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)


Para cargar y probar el modelo:
model = load_model('modelo_clasificacion_mes.keras')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)



🏆 Conclusiones
- Los atributos físicos del objeto tienen correlación con el mes en que se clasifica.
- El uso de normalización y técnicas de Dropout ayuda a mejorar la precisión.
- El modelo logra una clasificación precisa y se puede mejorar con más ajustes en los hiperparámetros.
