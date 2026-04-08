"""Main API class for NFC e-ink card communication."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from nfc_eink.device import DeviceInfo, parse_device_info
from nfc_eink.exceptions import (
    AuthenticationError,
    CommunicationError,
    NfcEinkError,
    StatusWordError,
)
from nfc_eink.image import encode_image
from nfc_eink.protocol import (
    build_auth_apdu,
    build_device_info_apdu,
    build_poll_apdu,
    build_refresh_apdu,
    is_refresh_complete,
)

if TYPE_CHECKING:
    from PIL import Image


class EInkCard:
    """NFC e-ink card communication manager.

    On connect, the card is automatically authenticated and device info
    is read, making serial_number and device_info immediately available.

    Usage::

        with EInkCard() as card:
            print(card.serial_number)
            card.send_image(Image.open("photo.png"))
            card.refresh()
    """

    _DELAY_S = 0.250

    def __init__(self, connection: Any = None) -> None:
        """Initialize EInkCard.

        Args:
            connection: A pyscard CardConnection object. If None, call connect()
                to auto-detect a reader and wait for a card.
        """
        self._connection = connection
        self._device_info: DeviceInfo | None = None

    @property
    def device_info(self) -> DeviceInfo | None:
        """Detected device information, available after connect."""
        return self._device_info

    @property
    def serial_number(self) -> str:
        """Per-device identifier string from C0 tag, available after connect."""
        if self._device_info is None:
            return ""
        return self._device_info.serial_number

    def connect(self, reader: str = "PaSoRi") -> None:
        """Connect to an NFC reader, wait for a card, authenticate, and read device info.

        Blocks until a card is detected on the reader.

        Args:
            reader: Name substring to match against available PC/SC readers
                (default: 'PaSoRi'). The first reader whose name contains this
                string is used.

        Raises:
            CommunicationError: If no matching reader is found or connection fails.
        """
        try:
            from smartcard.System import readers as get_readers
        except ImportError as e:
            raise CommunicationError(
                "pyscard is required: pip install pyscard"
            ) from e

        try:
            available = get_readers()
        except Exception as e:
            raise CommunicationError(f"Cannot access PC/SC readers: {e}") from e

        if not available:
            raise CommunicationError("No NFC readers found")

        matches = [r for r in available if reader in str(r)]
        if not matches:
            names = ", ".join(f'"{r}"' for r in available)
            raise CommunicationError(
                f"No reader matching '{reader}' found. Available: {names}"
            )
        selected = matches[0]

        while True:
            try:
                connection = selected.createConnection()
                connection.connect()
            except Exception:
                time.sleep(EInkCard._DELAY_S)
                continue
            self._connection = connection
            try:
                self.authenticate()
                break
            except AuthenticationError as e:
                self._connection = None
                connection.disconnect()
                time.sleep(EInkCard._DELAY_S)
                continue

        self._read_device_info()

    def _read_device_info(self) -> None:
        """Read and parse device info from the card."""
        cla, ins, p1, p2, data = build_device_info_apdu()
        raw = self._send_apdu(cla, ins, p1, p2, data, mrl=256)
        self._device_info = parse_device_info(raw)

    def close(self) -> None:
        """Close the NFC connection."""
        if self._connection is not None:
            try:
                self._connection.disconnect()
            except Exception:
                pass
            self._connection = None

    def __enter__(self) -> EInkCard:
        if self._connection is None:
            self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        self.close()

    def _send_apdu(
        self,
        cla: int,
        ins: int,
        p1: int,
        p2: int,
        data: bytes | None = None,
        mrl: int = 0,
        check_status: bool = True,
    ) -> bytes:
        """Send an APDU command via pyscard."""
        if self._connection is None:
            raise CommunicationError("Not connected to a card")

        apdu = [cla, ins, p1, p2]
        if data:
            apdu += [len(data)] + list(data)
        if mrl:
            apdu += [mrl & 0xFF]

        try:
            response, sw1, sw2 = self._connection.transmit(apdu)
        except Exception as e:
            raise CommunicationError(f"APDU command failed: {e}") from e

        if check_status and (sw1, sw2) != (0x90, 0x00):
            raise StatusWordError(sw1, sw2)

        return bytes(response)

    def authenticate(self) -> None:
        """Authenticate with the card.

        Raises:
            AuthenticationError: If authentication fails.
        """
        cla, ins, p1, p2, data = build_auth_apdu()
        try:
            self._send_apdu(cla, ins, p1, p2, data)
        except (StatusWordError, CommunicationError) as e:
            raise AuthenticationError(f"Authentication failed: {e}") from e

    def send_image(
        self,
        image: Any,
        dither: str = "pillow",
        resize: str = "fit",
        palette: str = "pure",
        tone_map: bool | None = None,
    ) -> None:
        """Send an image to the card.

        Accepts either a PIL Image (requires Pillow) or a 2D list of
        color indices matching the device's screen dimensions.
        Image encoding parameters are automatically determined from device info.

        Args:
            image: PIL Image or 2D list of color indices.
            dither: Dithering algorithm for PIL Image conversion.
                One of 'pillow' (default), 'atkinson', 'floyd-steinberg',
                'jarvis', 'stucki', 'none'.
            resize: Resize mode for PIL Image conversion.
                'fit' (default) adds white margins, 'cover' crops excess.
            palette: Palette mode for PIL Image conversion.
                'pure' (default) uses ideal RGB values.
                'tuned' uses colors adjusted for actual panel appearance.
            tone_map: Enable luminance tone mapping. None (default)
                enables it automatically for 'tuned' palette.

        Raises:
            CommunicationError: If sending fails.
            NfcEinkError: If image format is invalid.
        """
        is_pil = False
        try:
            from PIL import Image as PILImage

            is_pil = isinstance(image, PILImage.Image)
        except ImportError:
            pass

        if is_pil:
            from nfc_eink.convert import convert_image

            di = self._device_info
            if di is not None:
                pixels = convert_image(
                    image, di.width, di.height, di.num_colors,
                    dither=dither, resize=resize, palette=palette,
                    tone_map=tone_map,
                )
            else:
                pixels = convert_image(
                    image, dither=dither, resize=resize, palette=palette,
                    tone_map=tone_map,
                )
        else:
            pixels = image

        apdus = encode_image(pixels, self._device_info)

        for block_apdus in apdus:
            for cla, ins, p1, p2, data in block_apdus:
                self._send_apdu(cla, ins, p1, p2, data)

    def refresh(self, timeout: float = 30.0, poll_interval: float = 0.5) -> None:
        """Start screen refresh and wait for completion.

        Args:
            timeout: Maximum seconds to wait for refresh (default 30).
            poll_interval: Seconds between poll attempts (default 0.5).

        Raises:
            NfcEinkError: If refresh times out.
            CommunicationError: If communication fails during refresh.
        """
        cla, ins, p1, p2, data = build_refresh_apdu()
        self._send_apdu(cla, ins, p1, p2, data, mrl=256)

        cla, ins, p1, p2, data = build_poll_apdu()
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            response = self._send_apdu(cla, ins, p1, p2, data, mrl=1, check_status=False)
            if is_refresh_complete(response):
                return
            time.sleep(poll_interval)

        raise NfcEinkError(f"Screen refresh timed out after {timeout}s")
