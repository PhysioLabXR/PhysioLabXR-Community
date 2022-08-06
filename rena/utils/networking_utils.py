import zmq


def send_string_router_dealer(message, routing_id, socket_interface):
    socket_interface.socket.send_multipart(
        [routing_id, message.encode('utf-8')])


def send_router_dealer(data, routing_id, socket_interface):
    socket_interface.socket.send_multipart(
        [routing_id, data])


def recv_string_router_dealer(socket_interface, is_block):
    if is_block:
        routing_id, message = socket_interface.socket.recv_multipart(flags=0)
        return message.decode('utf-8'), routing_id
    else:
        try:
            routing_id, message = socket_interface.socket.recv_multipart(
                flags=zmq.NOBLOCK)
            return message.decode('utf-8'), routing_id
        except zmq.error.Again:
            return None  # no message has arrived at the socket yet
