import mysql.connector

def connect_to_db():
    try:
        connection = mysql.connector.connect(
            host="localhost",       # Dirección del servidor
            port=3307,              # Puerto de MySQL
            user="root",            # Usuario de MySQL
            password="",            # Contraseña del usuario
            database="productos"    # Base de datos a conectar
        )
        print("Conexión exitosa a la base de datos")
        return connection
    except mysql.connector.Error as err:
        print(f"Error al conectar a la base de datos: {err}")
        return None