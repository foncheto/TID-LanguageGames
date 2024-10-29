import random

# Definir los objetos en el "mundo" y los conceptos asociados
objetos = ['círculo', 'cuadrado', 'triángulo']
conceptos = ['forma redonda', 'forma con cuatro lados', 'forma triangular']

# Clase para el agente básico
class Agente:
    def __init__(self, nombre):
        self.nombre = nombre
        # Conocimiento inicial de asociaciones percepción-concepto
        self.conocimiento = {obj: concepto for obj, concepto in zip(objetos, conceptos)}
        # Memoria adaptativa para almacenar asociaciones incorrectas
        self.memoria_adaptativa = []

    def percibir(self):
        # Selecciona un objeto del "mundo" y lo percibe
        objeto_percibido = random.choice(objetos)
        print(f"{self.nombre} percibe el objeto: {objeto_percibido}")
        return objeto_percibido

    def conceptualizar(self, objeto):
        # Conceptualiza el objeto percibido usando su conocimiento
        concepto = self.conocimiento.get(objeto, "concepto desconocido")
        print(f"{self.nombre} conceptualiza el objeto '{objeto}' como: {concepto}")
        return concepto

    def interpretar(self, concepto):
        # Intenta deducir el objeto a partir del concepto recibido
        print(f"{self.nombre} intenta interpretar el concepto: {concepto}")
        # Evita asociaciones que están en la memoria adaptativa
        for obj, concepto_almacenado in self.conocimiento.items():
            if concepto_almacenado == concepto and (obj, concepto) not in self.memoria_adaptativa:
                print(f"{self.nombre} deduce que el objeto es: {obj}")
                return obj
        return "objeto desconocido"

    def actualizar_memoria_adaptativa(self, objeto, concepto, exito):
        # Almacena asociaciones incorrectas en la memoria adaptativa
        if not exito:
            if (objeto, concepto) not in self.memoria_adaptativa:
                print(f"{self.nombre} agrega a memoria adaptativa: ({objeto}, {concepto})")
                self.memoria_adaptativa.append((objeto, concepto))
            # Limitar el tamaño de la memoria adaptativa para mantener la eficiencia
            if len(self.memoria_adaptativa) > 5:
                self.memoria_adaptativa.pop(0)

# Crear los agentes
agente1 = Agente("Agente1")
agente2 = Agente("Agente2")

# Definir el ciclo de interacción entre los agentes
def ciclo_interaccion(agente_emisor, agente_receptor):
    # Paso 1: Percepción y Conceptualización
    objeto_percibido = agente_emisor.percibir()
    concepto = agente_emisor.conceptualizar(objeto_percibido)

    # Paso 2: Interpretación por el receptor
    objeto_deducido = agente_receptor.interpretar(concepto)
    
    # Paso 3: Retroalimentación y actualización de memoria adaptativa
    exito = objeto_deducido == objeto_percibido
    print(f"¿Éxito en la deducción? {'Sí' if exito else 'No'}\n")
    
    # Ambos agentes actualizan sus memorias adaptativas según el éxito de la interpretación
    agente_emisor.actualizar_memoria_adaptativa(objeto_percibido, concepto, exito)
    agente_receptor.actualizar_memoria_adaptativa(objeto_percibido, concepto, exito)

# Realizar varias interacciones
for i in range(5):
    print(f"\n--- Interacción {i+1} ---")
    ciclo_interaccion(agente1, agente2)
    ciclo_interaccion(agente2, agente1)
