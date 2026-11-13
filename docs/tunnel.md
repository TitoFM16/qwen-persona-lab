# Reconnecting local UI tunnel

Run the tunnel from the Mac, after the two remote servers are listening on their own loopback ports:

```bash
bash scripts/tunnel.sh SSH_HOST SSH_PORT .state/known_hosts
```

Open `http://127.0.0.1:8787` for the lab and `http://127.0.0.1:8791` for the original Qwen3.5 explorer. The script explicitly binds both local forwards to `127.0.0.1`, uses key-based batch authentication, and keeps strict host-key verification enabled. No model credentials or account tokens are stored by the script.

It sends SSH keepalives every 30 seconds and abandons an unresponsive connection after three unanswered keepalives. When SSH exits, the script waits three seconds and reconnects. Press **Ctrl+C** in its Terminal or PTY to terminate the script and its own SSH child. It never kills other tunnels, servers, or applications.

If a previous tunnel already occupies either local port, the script reports the forwarding error and exits. Close your old tunnel in its own Terminal, or select unused local ports:

```bash
LAB_LOCAL_PORT=8788 EXPLORER_LOCAL_PORT=8792 \
  bash scripts/tunnel.sh SSH_HOST SSH_PORT .state/known_hosts
```

Those values change only the Mac-side ports; the remote destinations remain `127.0.0.1:8787` and `127.0.0.1:8791`. The lab's strict HTTP Host check expects its configured server port, so use the default lab local port for ordinary browser access unless you have deliberately configured matching server/Host handling. An alternate explorer port likewise needs matching upstream Host handling. Alternate ports can help diagnose conflicts but do not automatically relax either server's Host protection.

The known-hosts file must already contain a verified server key. If it is missing or the host key has changed, investigate and verify the fingerprint through your provider. The helper does not automatically accept or replace keys.

## Direct-address host-key alias

The current deployment's verified key was recorded under `[ssh5.vast.ai]:23520`, while the direct SSH endpoint is `71.249.146.155:20322`. Use the verified alias explicitly when connecting directly:

```bash
SSH_HOST_KEY_ALIAS='[ssh5.vast.ai]:23520' \
  bash scripts/tunnel.sh 71.249.146.155 20322 .state/known_hosts
```

`HostKeyAlias` changes only the known-hosts lookup name. It does not change the server key, bypass fingerprint verification, or redirect the connection. Endpoint addresses can change across rentals; inspect the active provider instance before reusing these values.

If a server process is stopped remotely, a healthy SSH tunnel can still exist but its forwarded HTTP requests will fail. Restart the corresponding remote server rather than opening more tunnels. These two servers have separate model runtimes; avoid loading multiple model copies during worker inference.
