#!/usr/bin/env bash
# Print the Mac's IP on the iPhone USB / Personal Hotspot network (172.20.10.x), else en0,
# plus the exact Sensor Logger HTTP Push URL.
PORT="${1:-8000}"
IP=""; IFACE=""
while read -r iface ip; do
  case "$ip" in 172.20.10.*) IP="$ip"; IFACE="$iface"; break;; esac
done < <(ifconfig | awk '/^[a-z]/{i=$1; sub(":","",i)} /inet /{print i, $2}')
if [ -z "$IP" ]; then IP="$(ipconfig getifaddr en0 2>/dev/null)"; IFACE="en0"; fi
if [ -z "$IP" ]; then echo "No network IP found. Is the iPhone USB hotspot on?" >&2; exit 1; fi
case "$IP" in 172.20.10.*) NOTE="iPhone USB/hotspot";; *) NOTE="fallback, not the iPhone network";; esac
echo "Mac IP: $IP ($IFACE, $NOTE)"
echo "Sensor Logger HTTP Push URL: http://$IP:$PORT/data"
