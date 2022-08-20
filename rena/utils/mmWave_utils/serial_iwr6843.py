import time


def serial_config(config_fn, cli_port):
    # Open the serial ports for the configuration and the data ports

    # Read the configuration file and send it to the board
    try:
        config = [line.rstrip('\r\n') for line in open(config_fn)]
    except FileNotFoundError:
        print('Config file not found')
    for line in config:
        cli_port.write((line + '\r').encode())
        time.sleep(0.01)

    time.sleep(1)
    cli_result = cli_port.read(cli_port.in_waiting).decode()
    print(cli_result)  # CLI output of the board


def sensor_start(cli_port):
    cli_port.write(('sensorStart 0\n').encode())
    result = cli_port.read(cli_port.in_waiting).decode()

    print(result)


def sensor_stop(cli_port):
    cli_port.write(('sensorStop\n').encode())
    result = cli_port.read(cli_port.in_waiting).decode()

    print(result)


def close_connection(user_port, data_port):
    user_port.reset_input_buffer()
    user_port.reset_output_buffer()

    data_port.reset_input_buffer()
    data_port.reset_output_buffer()

    user_port.close()
    data_port.close()


def clear_serial_buffer(user_port, data_port):
    user_port.reset_input_buffer()
    user_port.reset_output_buffer()

    data_port.reset_input_buffer()
    data_port.reset_output_buffer()
