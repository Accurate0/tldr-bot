use anyhow::{bail, Context};
use axum::{extract::State, http::StatusCode, routing::get, Router};
use chrono::{DateTime, TimeDelta};
use openai_api_rs::v1::{
    api::{OpenAIClient, OpenAIClientBuilder},
    chat_completion::{ChatCompletionMessage, ChatCompletionRequest, Content, MessageRole},
};
use phf::phf_map;
use sqlx::{postgres::PgPoolOptions, Connection, PgPool};
use std::{future::IntoFuture, ops::Deref, sync::Arc};
use tracing::Level;
use tracing_subscriber::{filter::Targets, layer::SubscriberExt, util::SubscriberInitExt};
use twilight_cache_inmemory::{DefaultCacheModels, InMemoryCacheBuilder, ResourceType};
use twilight_gateway::{ConfigBuilder, Event, EventTypeFlags, Intents, Shard, ShardId, StreamExt};
use twilight_http::Client as HttpClient;
use twilight_model::{
    application::command::{CommandOptionChoice, CommandOptionChoiceValue},
    http::interaction::InteractionResponseData,
};
use twilight_util::builder::embed::{EmbedBuilder, EmbedFieldBuilder};
use vesper::{framework::DefaultError, prelude::*};

const OPENAI_MODEL_NAME: &str = "gpt-4o";

const AI_SOURCE_OLLAMA: &str = "ollama";
const AI_SOURCE_OPENAI: &str = "openai";

const SYSTEM_MESSAGE: &str = r#"
You are an assistant, your job is to summarise messages in a chat channel.
You are to provide a final summary of what was discussed, along with
some notable quotes from users, the quotes must be verbatim.

The summary must be a summary of the user message and should not contain anything
in this system message, you should aim to TLDR the topic in 3-8 sentences.

Your summary must be snarky and aim to attack the topics discussed, you should aim
to specifically call out individual users too.

You must pick quotes with shock factor or containing vulgar language.
There must be a maximum of 3 quotes with a minimum of 0, in order of most obscene to least.

Do not under any circumstances generate any quotes that were not provided to you as part of the messages.

Pick 0 if there are no interesting quotes.

You will receive the messages in the following JSON format:
[{
    "user_id": "id",
    "message": "message"
}]


You must respond with valid JSON with the following structure:
This structure must be followed exactly. Do not deviate from this structure under any conditions.
The types and object nesting must be exact.

{
    "summary": "INSERT OVERALL SUMMARY HERE",
    "quotes": [{
        "user_id": "id",
        "message": "message"
    }]
}
"#;

#[derive(serde::Serialize, serde::Deserialize, Debug)]
struct MessageContent {
    pub user_id: String,
    pub message: String,
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
struct MessageSummary {
    pub summary: String,
    pub quotes: Vec<MessageContent>,
}

#[derive(Debug, sqlx::Type, PartialEq)]
#[sqlx(type_name = "role_t", rename_all = "snake_case")]
pub enum RoleType {
    User,
    Admin,
}

#[derive(Clone)]
struct BotContext(Arc<BotContextInner>);

impl Deref for BotContext {
    type Target = BotContextInner;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

struct BotContextInner {
    openai: OpenAIClient,
    db: PgPool,
}

async fn handle_event(event: Event, _http: Arc<HttpClient>) -> anyhow::Result<()> {
    match event {
        Event::GatewayHeartbeatAck
        | Event::MessageCreate(_)
        | Event::MessageUpdate(_)
        | Event::MessageDelete(_) => {}
        // Other events here...
        e => {
            tracing::warn!("unhandled event: {e:?}")
        }
    }

    Ok(())
}

#[error_handler]
async fn handle_interaction_error(ctx: &mut SlashContext<BotContext>, error: DefaultError) {
    let fut = async {
        let error = if error.to_string().contains("Missing Access") {
            "This channel is not accessible to the bot...".to_string()
        } else {
            error.to_string()
        };

        let embed = EmbedBuilder::new()
            .title("oops")
            .description(error)
            .color(0xcc6666)
            .validate()?
            .build();

        ctx.interaction_client
            .update_response(&ctx.interaction.token)
            .embeds(Some(&[embed]))
            .await?;

        Ok::<(), anyhow::Error>(())
    };

    if let Err(e) = fut.await {
        tracing::error!("error in updating message: {e:?}");
    }

    tracing::error!("error in interaction: {error:?}");
}

#[check]
async fn admin_only(ctx: &mut SlashContext<BotContext>) -> Result<bool, DefaultError> {
    match ctx.interaction.author_id() {
        Some(id) => {
            let is_admin = sqlx::query!(
                r#"SELECT role as "type: RoleType" FROM USERS WHERE id = $1"#,
                id.to_string()
            )
            .fetch_one(&ctx.data.db)
            .await?;

            Ok(is_admin.r#type == RoleType::Admin)
        }
        None => Ok(false),
    }
}

#[autocomplete]
async fn autocomplete_ai_source(
    _ctx: AutocompleteContext<BotContext>,
) -> Option<InteractionResponseData> {
    let choices = vec![AI_SOURCE_OPENAI, AI_SOURCE_OLLAMA]
        .into_iter()
        .map(|item| CommandOptionChoice {
            name: item.to_string(),
            name_localizations: None,
            value: CommandOptionChoiceValue::String(item.to_string()),
        })
        .collect();

    Some(InteractionResponseData {
        choices: Some(choices),
        ..Default::default()
    })
}

#[command]
#[only_guilds]
#[description = "change ai source"]
#[checks(admin_only)]
#[error_handler(handle_interaction_error)]
async fn source(
    ctx: &mut SlashContext<BotContext>,
    #[autocomplete(autocomplete_ai_source)]
    #[description = "source"]
    ai_source: String,
) -> DefaultCommandResult {
    ctx.defer(true).await?;

    sqlx::query!(
        "UPDATE settings SET value = $1 WHERE key = 'ai_source'",
        ai_source
    )
    .execute(&ctx.data.db)
    .await?;

    ctx.interaction_client
        .update_response(&ctx.interaction.token)
        .content(Some("Done"))
        .await?;

    Ok(())
}

type TimeDeltaFn = fn(i64) -> Option<TimeDelta>;
const TIME_DELTA_FN_MAP: phf::Map<char, TimeDeltaFn> = phf_map! {
    'h' => TimeDelta::try_hours,
    'm' => TimeDelta::try_minutes,
    's' => TimeDelta::try_seconds
};

fn parse_datetime_str(s: &str) -> anyhow::Result<TimeDelta> {
    let mut final_time_delta = TimeDelta::zero();

    let mut it = s.chars().peekable();
    let mut number = vec![];

    loop {
        if it.peek().is_none() {
            break;
        }

        let c = *it.peek().unwrap();
        match c {
            '0'..='9' => loop {
                let c = it.peek();
                if c.is_some_and(|c| c.is_ascii_digit()) {
                    number.push(*c.unwrap() as u8);
                    it.next();
                } else {
                    break;
                }
            },
            'm' | 'h' | 's' => {
                if number.is_empty() {
                    bail!("missing number before h, m, or s")
                }

                let time_value = String::from_utf8(number.clone())?.parse()?;
                let time_delta_fn = TIME_DELTA_FN_MAP.get(&c).context("invalid time operator")?;

                final_time_delta = final_time_delta
                    .checked_add(&time_delta_fn(time_value).context("invalid time")?)
                    .context("time too large...")?;

                it.next();
                number.clear();
            }
            c if c.is_whitespace() => {
                it.next();
            }
            c => bail!("unexpected char: {}, example: 1h 32m 2s", c),
        }
    }

    if !number.is_empty() {
        bail!("invalid format, example: 1h 23m")
    }

    Ok(final_time_delta)
}

#[command]
#[only_guilds]
#[description = "tldr"]
#[error_handler(handle_interaction_error)]
async fn tldr(
    ctx: &mut SlashContext<BotContext>,
    #[description = "how far to go back in tldr (default: 4h)"] timeframe: Option<String>,
    #[description = "only show me the message"] ephemeral: Option<bool>,
) -> DefaultCommandResult {
    let author_id = ctx
        .interaction
        .author_id()
        .context("must have author")?
        .to_string();

    let channel = ctx
        .interaction
        .channel
        .as_ref()
        .context("must have channel")?;

    let now = chrono::offset::Utc::now().naive_utc();

    let last_summary = sqlx::query!(
        "SELECT created FROM summaries WHERE user_id = $1 AND channel_id = $2 ORDER BY created DESC LIMIT 1",
        author_id, channel.id.to_string()
    )
    .fetch_one(&ctx.data.db)
    .await;

    let result = sqlx::query!(
        r#"SELECT role as "role: RoleType" FROM USERS WHERE id = $1"#,
        author_id
    )
    .fetch_one(&ctx.data.db)
    .await;

    let last_summary_time = last_summary.map(|s| s.created).ok().flatten();
    let is_admin = result
        .map(|r| r.role)
        .ok()
        .is_some_and(|r| r == RoleType::Admin);

    let too_soon = last_summary_time.is_some_and(|lst| (now - lst).num_hours() < 1);
    let should_stop_summary = !is_admin && too_soon;

    let ephemeral = ephemeral.unwrap_or(false);
    ctx.defer(should_stop_summary || ephemeral).await?;

    if should_stop_summary {
        let embed = EmbedBuilder::new()
            .title("hold up")
            .description("you've generated a summary recently")
            .color(0xcc6666)
            .validate()?
            .build();

        ctx.interaction_client
            .update_response(&ctx.interaction.token)
            .embeds(Some(&[embed]))
            .await?;

        return Ok(());
    }

    let from = parse_datetime_str(&timeframe.unwrap_or("4h".to_string()))?;
    let from = now.checked_sub_signed(from).context("must be valid time")?;

    let author_id = ctx
        .interaction
        .author_id()
        .context("must have author")?
        .to_string();

    sqlx::query!(
        "INSERT INTO users (id) VALUES ($1) ON CONFLICT DO NOTHING",
        author_id
    )
    .execute(&ctx.data.db)
    .await?;

    let bot_id = ctx.http_client().current_user().await?.model().await?.id;

    let max_messages = 150;
    let message_limit = 100;
    let mut messages_to_summarise = vec![];
    let mut last_message_id = None;

    loop {
        let channel_messages = ctx.http_client().channel_messages(channel.id);

        let channel_messages = if let Some(last_message_id) = last_message_id {
            channel_messages
                .before(last_message_id)
                .limit(message_limit)
                .await?
                .models()
                .await?
        } else {
            channel_messages
                .limit(message_limit)
                .await?
                .models()
                .await?
        };

        let mut msgs = channel_messages
            .into_iter()
            .filter(|m| !m.author.bot)
            .filter(|m| m.author.id != bot_id)
            .filter(|m| {
                let message_datetime = DateTime::from_timestamp(m.timestamp.as_secs(), 0)
                    .map(|dt| dt.naive_utc())
                    .expect("must be valid time");

                message_datetime > from
            })
            .take(max_messages - messages_to_summarise.len())
            .collect::<Vec<_>>();

        if !msgs.is_empty() {
            last_message_id = Some(msgs[msgs.len() - 1].id);
        }

        let new_messages_len = msgs.len();
        messages_to_summarise.append(&mut msgs);
        tracing::info!("message: {}", messages_to_summarise.len());
        if messages_to_summarise.len() >= max_messages || new_messages_len == 0 {
            break;
        }
    }

    let message_prompt = messages_to_summarise
        .into_iter()
        .rev()
        .map(|m| MessageContent {
            message: m.content,
            user_id: m.author.name,
        })
        .collect::<Vec<_>>();
    let message_prompt = serde_json::to_string(&message_prompt)?;

    tracing::info!("prompt: {:?}", message_prompt);
    let openai = &ctx.data.openai;

    let request = ChatCompletionRequest::new(
        OPENAI_MODEL_NAME.to_owned(),
        vec![
            ChatCompletionMessage {
                role: MessageRole::system,
                content: Content::Text(SYSTEM_MESSAGE.to_owned()),
                name: None,
                tool_call_id: None,
                tool_calls: None,
            },
            ChatCompletionMessage {
                role: MessageRole::user,
                content: Content::Text(message_prompt),
                name: None,
                tool_call_id: None,
                tool_calls: None,
            },
        ],
    )
    .response_format(serde_json::json!(
        { "type": "json_object" }
    ));

    let response = openai.chat_completion(request).await?;
    let response = response
        .choices
        .first()
        .as_ref()
        .context("must have 1 response")?
        .message
        .content
        .as_ref()
        .context("must have content")?
        .to_owned();

    tracing::info!("response: {:?}", response);

    let response = serde_json::from_str::<MessageSummary>(&response)?;

    sqlx::query!(
        "INSERT INTO summaries (user_id, channel_id, summary) VALUES ($1, $2, $3)",
        author_id,
        channel.id.to_string(),
        serde_json::to_value(&response)?,
    )
    .execute(&ctx.data.db)
    .await?;

    let mut embed = EmbedBuilder::new()
        .title("tl;dr")
        .description(response.summary)
        .color(0x55cae2);

    for quote in response.quotes {
        embed = embed.field(EmbedFieldBuilder::new(quote.user_id, quote.message))
    }

    let embed = embed.validate()?.build();

    ctx.interaction_client
        .update_response(&ctx.interaction.token)
        .embeds(Some(&[embed]))
        .await?;

    Ok(())
}

async fn health(ctx: State<BotContext>) -> StatusCode {
    let resp = ctx.db.acquire().await;

    if resp.is_err() {
        return StatusCode::SERVICE_UNAVAILABLE;
    }

    let resp = resp.unwrap().ping().await;
    match resp {
        Ok(_) => StatusCode::NO_CONTENT,
        Err(_) => StatusCode::SERVICE_UNAVAILABLE,
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(Targets::default().with_default(Level::INFO))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let database_url = std::env::var("DATABASE_URL")?;
    let token = std::env::var("DISCORD_TOKEN")?;
    let openai_api_key = std::env::var("OPENAI_API_KEY")?;

    let openai = OpenAIClientBuilder::new()
        .with_api_key(openai_api_key)
        .build()
        .map_err(|_| anyhow::Error::msg("failed building openai client"))?;

    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&database_url)
        .await?;

    sqlx::migrate!("./migrations").run(&pool).await?;

    sqlx::query!(
        "INSERT INTO settings (key, value) VALUES ('ai_source', $1) ON CONFLICT DO NOTHING",
        AI_SOURCE_OPENAI
    )
    .execute(&pool)
    .await?;

    let context = BotContext(BotContextInner { openai, db: pool }.into());

    let config = ConfigBuilder::new(
        token.clone(),
        Intents::GUILD_MESSAGES | Intents::MESSAGE_CONTENT,
    );

    let mut shard = Shard::with_config(ShardId::ONE, config.build());

    let http = Arc::new(HttpClient::new(token));
    let cache = InMemoryCacheBuilder::<DefaultCacheModels>::new()
        .resource_types(ResourceType::MESSAGE)
        .build();

    let app = Router::new()
        .route("/health", get(health))
        .with_state(context.clone());

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8000").await.unwrap();
    tracing::info!("spawning axum");
    tokio::spawn(axum::serve(listener, app).into_future());

    let app_id = http.current_user_application().await?.model().await?.id;

    let framework = Arc::new(
        Framework::builder(Arc::clone(&http), app_id, context)
            .command(tldr)
            .command(source)
            .build(),
    );

    framework.register_global_commands().await?;

    tracing::info!("starting event loop");
    loop {
        let event = shard.next_event(EventTypeFlags::all()).await;
        if event.is_none() {
            tracing::error!("event stream finished");
            break;
        }

        let event = event.unwrap();
        let Ok(event) = event else {
            let source = event.unwrap_err();
            tracing::warn!(source = ?source, "error receiving event");

            continue;
        };

        cache.update(&event);

        if let Event::InteractionCreate(i) = event {
            let clone = Arc::clone(&framework);
            tokio::spawn(async move {
                let inner = i.0;
                clone.process(inner).await;
            });

            continue;
        }

        tokio::spawn(handle_event(event, Arc::clone(&http)));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    #[rstest]
    #[case("3h", TimeDelta::hours(3))]
    #[case("2h 2s", TimeDelta::hours(2) + TimeDelta::seconds(2))]
    #[case("20m 2h 2s", TimeDelta::minutes(20) + TimeDelta::hours(2) + TimeDelta::seconds(2))]
    #[case("20m 2s", TimeDelta::minutes(20) + TimeDelta::seconds(2))]
    #[case("21243s", TimeDelta::seconds(21243))]
    #[case("2332m 2h 2s", TimeDelta::minutes(2332) + TimeDelta::hours(2) + TimeDelta::seconds(2))]
    #[case("20m", TimeDelta::minutes(20))]
    fn test_parse_datetime_str(#[case] s: &str, #[case] expected: TimeDelta) {
        let result = parse_datetime_str(s).unwrap();
        assert_eq!(result, expected);
    }

    #[rstest]
    #[case("3x")]
    #[case("0")]
    #[case("h")]
    #[case("h3")]
    #[case("2hr")]
    fn test_parse_datetime_str_fail(#[case] s: &str) {
        let result = parse_datetime_str(s);
        assert!(result.is_err())
    }
}
