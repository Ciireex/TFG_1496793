# 🧠 TFG_RL – Entorno de Aprendizaje por Refuerzo para un Juego de Estrategia por Turnos

Este proyecto forma parte del Trabajo de Fin de Grado (TFG) en Ingeniería Informática en la Universitat Autònoma de Barcelona. Consiste en el diseño e implementación de un entorno compatible con Gymnasium para entrenar agentes de **aprendizaje por refuerzo (RL)** en un juego de estrategia por turnos.

## 🎯 Objetivos del Proyecto

- Diseñar un entorno estratégico con reglas claras y mecánicas tácticas inspiradas en *Advance Wars* y *Fire Emblem*.
- Integrar y comparar distintos algoritmos de RL (PPO, A2C, DQN y Maskable PPO) usando **Stable-Baselines3**.
- Evaluar la efectividad de cada algoritmo en entornos progresivamente más complejos.
- Aplicar técnicas como *transfer learning*, *reward shaping* y redes convolucionales personalizadas para mejorar el rendimiento.

## 🕹️ Mecánicas del Juego

- Tablero por casillas con terreno especial (bosques, colinas, campamentos, obstáculos).
- Tres tipos de unidades: **Soldado**, **Arquero** y **Caballero**, con sistema de debilidades en triángulo.
- Turnos alternos: cada unidad realiza una fase de movimiento y otra de ataque.
- Objetivo: **eliminar todas las unidades del equipo rival**.

## 🧠 Arquitectura del Entorno

- Compatible con la API estándar de Gymnasium (`reset()` y `step()`).
- Observaciones espaciales multicanal (21 canales): unidades, salud, terreno, turno, etc.
- Espacio de acción `Discrete(5)` que varía según la fase del turno.
- Penalización por acciones inválidas y recompensas por comportamiento estratégico.

## 🤖 Algoritmos Evaluados

Se entrenaron y compararon cuatro algoritmos:

- **PPO**
- **A2C**
- **DQN**
- **Maskable PPO**

El entrenamiento se realizó en **7 fases progresivas**, desde mapas simples con pocas unidades hasta mapas con terreno especial, obstáculos y múltiples tipos de unidad.

## 📊 Resultados

- **DQN Blue** fue el agente más eficaz globalmente.
- **Maskable PPO Blue** quedó muy cerca en rendimiento y destacó en consistencia.
- **PPO Red** empató muchas partidas, pero fue poco ofensivo.
- **A2C Red** obtuvo el peor desempeño.

Gráficas y tablas de resultados disponibles en las carpetas `/logs/` y `/graficas_tfevents/`.

## 👤 Autor

**Eric Rodríguez Merichal**  
📧 eric.rodriguez.merichal@e-campus.uab.cat

## 📅 Fecha de entrega

30 de junio de 2025
