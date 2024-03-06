from sshtunnel import SSHTunnelForwarder

server = SSHTunnelForwarder(
    'jackson.rovit.ua.es',
    ssh_username="rbustos",
    ssh_password="s0y3lr4f4+1",
    remote_bind_address=('127.0.0.1', 5000)
)

server.start()

print(server.local_bind_port)  # show assigned local port
# work with `SECRET SERVICE` through `server.local_bind_port`.

if input() == "c":
  server.stop()