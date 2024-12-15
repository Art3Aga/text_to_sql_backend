from db_connection import connect_to_db

connection = connect_to_db()
if connection:
    print("¡La conexión funciona correctamente!")
    connection.close()  # Cierra la conexión cuando termines
else:
    print("No se pudo conectar a la base de datos.")
