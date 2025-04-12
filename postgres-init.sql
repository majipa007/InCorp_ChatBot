CREATE TABLE chats
(
    id varchar PRIMARY KEY,
    name varchar NULL,
    email varchar NULL,
    phone varchar NULL,
    conversion boolean DEFAULT false,
    sentiment numeric NULL,
    last_updated timestamptz DEFAULT now(),
    created_at timestamptz DEFAULT now(),
    chat jsonb NOT NULL
);
