FROM pgvector/pgvector:pg17

RUN apt-get update && apt-get install -y \
  curl build-essential pkg-config libssl-dev git \
  clang-16 libclang-16-dev postgresql-server-dev-17 \
  && rm -rf /var/lib/apt/lists/*

ENV LIBCLANG_PATH=/usr/lib/llvm-16/lib \
  LD_LIBRARY_PATH=/usr/lib/llvm-16/lib

# Install Rust + pgrx
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"

# Install the correct version of cargo-pgrx
RUN cargo install --locked cargo-pgrx --version 0.16.1
RUN cargo pgrx init --pg17 $(which pg_config)

# Clone & install pgvector (if needed)
RUN cd /tmp \
  && git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git \
  && cd pgvector && make && make install

# Clone & install pgvectorscale
RUN cd /tmp \
  && git clone https://github.com/timescale/pgvectorscale.git \
  && cd pgvectorscale/pgvectorscale \
  && cargo pgrx install --release

# Clone & install pg_textsearch
RUN cd /tmp \
  && git clone https://github.com/timescale/pg_textsearch.git \
  && cd pg_textsearch \
  && make && make install
