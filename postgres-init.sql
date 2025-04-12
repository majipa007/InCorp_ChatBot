CREATE TABLE chats
(
    id varchar PRIMARY KEY,
    name varchar NULL,
    email varchar NULL,
    phone varchar NULL,
    sentiment numeric NULL,
    chat jsonb NOT NULL,
    last_updated timestamptz DEFAULT now(),
    created_at timestamptz DEFAULT now(),
    conversion boolean DEFAULT false
);
