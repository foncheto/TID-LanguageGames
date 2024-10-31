# Description: Juegos de lenguaje en Python
import random
import os

# Definir los objetos y las palabras asociadas
objetos = ['círculo', 'cuadrado', 'triángulo']
palabras = ['bla', 'blu', 'bli']

class Agente:
    def __init__(self):
        # El conocimiento es un diccionario que asocia palabras con probabilidades de objetos
        self.conocimiento = {palabra: {objeto: 1.0 / len(objetos) for objeto in objetos} for palabra in palabras}
        # Conocimiento inicial: {'bla': {'círculo': 0.33, 'cuadrado': 0.33, 'triángulo': 0.33}, 'blu': {'círculo': 0.33, 'cuadrado': 0.33, 'triángulo': 0.33}, 'bli': {'círculo': 0.33, 'cuadrado': 0.33, 'triángulo': 0.33}}

    def seleccionar_palabra(self, objeto):
        # Selecciona la palabra con mayor probabilidad para un objeto
        palabra_seleccionada = max(self.conocimiento, key=lambda p: self.conocimiento[p][objeto])
        return palabra_seleccionada

    def adivinar_objeto(self, palabra):
        # Adivina el objeto basado en las probabilidades asociadas a la palabra
        probabilidades = self.conocimiento[palabra]
        # {'círculo': 0.33, 'cuadrado': 0.33, 'triángulo': 0.33}
        objetos_posibles = list(probabilidades.keys())
        # ['círculo', 'cuadrado', 'triángulo']
        probabilidades_normalizadas = [probabilidades[obj] for obj in objetos_posibles]
        suma_probabilidades = sum(probabilidades_normalizadas)
        
        # Normalizar las probabilidades para que sumen 1
        probabilidades_normalizadas = [p / suma_probabilidades for p in probabilidades_normalizadas]
        
        # Elegir el objeto basado en las probabilidades
        objeto_adivinado = random.choices(objetos_posibles, weights=probabilidades_normalizadas, k=1)[0]
        # Objeto adivinado aleatoriamente basado en las probabilidades (puede no ser el correcto o no ser el más probable (diversidad), mientras más interacciones, más probable que sea el correcto)
        return objeto_adivinado

    def actualizar_conocimiento(self, palabra, objeto_correcto, exito):
        # Si es exitoso, refuerza la asociación entre la palabra y el objeto
        if exito:
            for objeto in self.conocimiento[palabra]:
                if objeto == objeto_correcto:
                    self.conocimiento[palabra][objeto] += 1.0  # Refuerza la conexión correcta
                else:
                    self.conocimiento[palabra][objeto] = max(0.1, self.conocimiento[palabra][objeto] - 0.1)  # Penaliza otras asociaciones (Minimo 0.1)
        else:
            # Si no es exitoso, debilita las conexiones con el objeto incorrecto
            self.conocimiento[palabra][objeto_correcto] += 0.5  # Incrementa la asociación correcta
            for objeto in self.conocimiento[palabra]:
                if objeto != objeto_correcto:
                    self.conocimiento[palabra][objeto] = max(0.1, self.conocimiento[palabra][objeto] - 0.1)  # Penaliza la incorrecta

# Realizar una interacción
def interaccion(agente_emisor, agente_receptor):
    # El emisor selecciona un objeto aleatorio y una palabra
    objeto_seleccionado = random.choice(objetos)
    palabra_seleccionada = agente_emisor.seleccionar_palabra(objeto_seleccionado)
    
    # El receptor intenta adivinar el objeto basado en la palabra
    objeto_adivinado = agente_receptor.adivinar_objeto(palabra_seleccionada)
    
    # Ver si el receptor adivinó correctamente
    exito = objeto_adivinado == objeto_seleccionado
    
    # Actualizar el conocimiento de ambos agentes
    agente_emisor.actualizar_conocimiento(palabra_seleccionada, objeto_seleccionado, exito)
    agente_receptor.actualizar_conocimiento(palabra_seleccionada, objeto_seleccionado, exito)
    
    # Imprimir el resultado de la interacción
    print(f"Emisor selecciona el objeto '{objeto_seleccionado}' y usa la palabra '{palabra_seleccionada}'.")
    print(f"Receptor adivina el objeto '{objeto_adivinado}'. {'Éxito' if exito else 'Fallo'}.")

# Inicializar dos agentes
agente1 = Agente()
agente2 = Agente()

# Simular varias interacciones
for i in range(10):
    interaccion(agente1, agente2)
    print("---")

"""print("------------------LLEGA OTRO AGENTE------------------")

# Llega otro agente como emisor luego de 10 interacciones
agente3 = Agente()

for i in range(10):
    interaccion(agente3, agente1)
    print("---")"""

print("------------------LLEGA OTRO AGENTE------------------")

# Llega otro agente como receptor luego de 10 interacciones
agente4 = Agente()

for i in range(10):
    interaccion(agente2, agente4)
    print("---")

