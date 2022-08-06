import zmq


def send_string_router_dealer(string, routing_id, socket_interface):
    socket_interface.socket.send_multipart(
        [routing_id, string.encode('utf-8')])


def send(data, routing_id, socket_interface):
    socket_interface.socket.send_multipart(
        [routing_id, data])


def recv_string(socket_interface, is_block):
    if is_block:
        routing_id, command = socket_interface.socket.recv_multipart(flags=0)
        return command.decode('utf-8'), routing_id
    else:
        try:
            routing_id, command = socket_interface.socket.recv_multipart(
                flags=zmq.NOBLOCK)
            return command.decode('utf-8'), routing_id
        except zmq.error.Again:
            return None  # no message has arrived at the socket yet