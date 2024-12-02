def calcular_total():
    total = 0
    while True:
        cantidad = input("Ingresa la cantidad (o 'no' para finalizar): ")
        if cantidad.lower() == "no":
            break
        try:
            cantidad = int(cantidad)
            precio_unitario = float(input("Ingresa el precio unitario: "))
            total += cantidad * precio_unitario
        except ValueError:
            print("Entrada inválida. Por favor, ingresa números válidos.")
            continue

    print(f"El total de la suma es: ${total:.2f}")

# Ejecutar la función
calcular_total()
