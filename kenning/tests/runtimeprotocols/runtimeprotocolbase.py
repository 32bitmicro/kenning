from kenning.runtimeprotocols.network import NetworkProtocol
from kenning.core.runtimeprotocol import RuntimeProtocol
import pytest


@pytest.mark.fast
class RuntimeProtocolTests:
    def initprotocol(self, *args, **kwargs) -> RuntimeProtocol:
        """
        Initializes protocol object.

        Returns
        -------
        RuntimeProtocol:
            Initialized protocol object
        """
        protocol = NetworkProtocol(self.host, self.port, *args, **kwargs)
        return protocol

    def test_initialize_server(self):
        server = self.initprotocol()
        assert server.initialize_server() is True
        with pytest.raises(OSError) as execinfo:
            second_server = self.initprotocol()
            second_server.initialize_server()
        assert 'Address already in use' in str(execinfo.value)
        server.disconnect()

    def test_initialize_client(self):
        client = self.initprotocol()
        with pytest.raises(ConnectionRefusedError):
            client.initialize_client()
        server = self.initprotocol()
        server.initialize_server()
        client.initialize_client()

        client.disconnect()
        server.disconnect()

    def test_wait_for_activity(self):
        raise NotImplementedError

    @pytest.mark.xfail()
    def test_send_data(self):
        assert 0

    @pytest.mark.xfail()
    def test_receive_data(self):
        assert 0

    @pytest.mark.xfail()
    def test_upload_input(self):
        assert 0

    @pytest.mark.xfail()
    def test_upload_model(self):
        assert 0

    @pytest.mark.xfail()
    def test_upload_quantization_details(self):
        assert 0

    @pytest.mark.xfail()
    def test_request_processing(self):
        assert 0

    @pytest.mark.xfail()
    def test_download_output(self):
        assert 0

    @pytest.mark.xfail()
    def test_download_statistics(self):
        assert 0

    @pytest.mark.xfail()
    def test_request_success(self):
        assert 0

    @pytest.mark.xfail()
    def test_request_failure(self):
        assert 0

    @pytest.mark.xfail()
    def test_parse_message(self):
        assert 0

    @pytest.mark.xfail()
    def test_disconnect(self):
        assert 0


@pytest.mark.fast
class TestCheckRequest:
    @pytest.mark.xfail()
    def test_one(self):
        assert 0


@pytest.mark.fast
class TestRequestFailure:
    @pytest.mark.xfail()
    def test_one(self):
        assert 0


@pytest.mark.fast
class TestMessageType:
    @pytest.mark.xfail()
    def test_to_bytes(self):
        assert 0

    @pytest.mark.xfail()
    def test_from_bytes(self):
        assert 0


@pytest.mark.fast
class TestServerStatus:
    @pytest.mark.xfail()
    def test_one(self):
        assert 0
